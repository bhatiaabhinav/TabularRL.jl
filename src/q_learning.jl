using MDPs
import MDPs: preexperiment, preepisode, prestep, poststep, postepisode, postexperiment
using UnPack
import DataStructures: CircularBuffer
using Random

export QLearner, QLearnerWithPriotiziedExperienceReplay, QLearnerWithExperienceReplay

"""Learns Q-Values for a given policy using the expected SARSA TD rule"""
mutable struct QLearner <: AbstractHook
    π::AbstractPolicy{Int, Int}
    q::Matrix{Float64}
    γ::Float64
    counts::Matrix{Int}
    s::Int
    QLearner(π::AbstractPolicy{Int, Int}, q::Matrix{Float64}, γ::Float64) = new(π, q, γ, zeros(Int, size(q)), 1)
end

function prestep(ql::QLearner; env::AbstractMDP, kwargs...)
    ql.s = state(env)
    nothing
end

function poststep(ql::QLearner; env::AbstractMDP, kwargs...)
    @unpack π, q, γ, counts, s = ql
    a, r, s′, d = action(env), reward(env), state(env), in_absorbing_state(env)
    δ = r + (1 - Float64(d)) * γ * sum(a′ -> π(s′, a′) * q[a′, s′], action_space(env)) - q[a, s]
    counts[a, s] += 1
    α = 1.0 / counts[a, s]
    q[a, s] += α * δ
    nothing
end


"""Learns Q-Values for a given policy using the expected SARSA TD rule and prioritized experience replay"""
mutable struct QLearnerWithPriotiziedExperienceReplay <: AbstractHook
    π::AbstractPolicy{Int, Int}
    q::Matrix{Float64}
    γ::Float64
    counts::Matrix{Int}
    s::Int
    db::Dict{Int, CircularBuffer{Tuple{Int, Int, Float64, Int, Bool}}}  # experiences leading to the key state
    QLearnerWithPriotiziedExperienceReplay(π::AbstractPolicy{Int, Int}, q::Matrix{Float64}, γ::Float64) = new(π, q, γ, zeros(Int, size(q)), 1, Dict{Int, CircularBuffer{Tuple{Int, Int, Float64, Int, Bool}}}())
end

function prestep(ql::QLearnerWithPriotiziedExperienceReplay; env::AbstractMDP, kwargs...)
    ql.s = state(env)
    if !haskey(ql.db, ql.s)
        ql.db[ql.s] = CircularBuffer{Tuple{Int, Int, Float64, Int, Bool}}(10000)
    end
    nothing
end

function poststep(ql::QLearnerWithPriotiziedExperienceReplay; env::AbstractMDP, rng::AbstractRNG, kwargs...)
    @unpack π, q, γ, counts, s, db = ql

    cur_val_s = sum(a -> π(s, a) * q[a, s], action_space(env))

    a, r, s′, d = action(env), reward(env), state(env), in_absorbing_state(env)
    δ = r + (1 - Float64(d)) * γ * sum(a′ -> π(s′, a′) * q[a′, s′], action_space(env)) - q[a, s]
    counts[a, s] += 1
    α = 1.0 / counts[a, s]
    q[a, s] += α * δ

    new_val_s = sum(a -> π(s, a) * q[a, s], action_space(env))

    if !haskey(db, s′)
        db[s′] = CircularBuffer{Tuple{Int, Int, Float64, Int, Bool}}(10000)
    end
    push!(db[s′], (s, a, r, s′, d))

    replay_predecessors_of = Set{Int}()

    if !isapprox(cur_val_s, new_val_s, rtol=1e-2)
        push!(replay_predecessors_of, s)
    end

    while !isempty(replay_predecessors_of)
        _s′ = pop!(replay_predecessors_of)
        for (s, a, r, s′, d) in Iterators.reverse(db[_s′])
            cur_val_s = sum(a -> π(s, a) * q[a, s], action_space(env))
            δ = r + (1 - Float64(d)) * γ * sum(a′ -> π(s′, a′) * q[a′, s′], action_space(env)) - q[a, s]
            α = 1.0 / counts[a, s]
            q[a, s] += α * δ
            new_val_s = sum(a -> π(s, a) * q[a, s], action_space(env))
            if !isapprox(cur_val_s, new_val_s, rtol=1e-2)
                push!(replay_predecessors_of, s)
            end
        end
    end
    nothing
end




"""Learns Q-Values for a given policy using the expected SARSA TD rule and experience replay"""
mutable struct QLearnerWithExperienceReplay <: AbstractHook
    π::AbstractPolicy{Int, Int}
    q::Matrix{Float64}
    γ::Float64
    counts::Matrix{Int}
    s::Int
    buffer::CircularBuffer{Tuple{Int, Int, Float64, Int, Bool}}
    batch_size::Int
    QLearnerWithExperienceReplay(π::AbstractPolicy{Int, Int}, q::Matrix{Float64}, γ::Float64; buffer_size=1000000, batch_size=32) = new(π, q, γ, zeros(Int, size(q)), 1, CircularBuffer{Tuple{Int, Int, Float64, Int, Bool}}(buffer_size), batch_size)
end

function prestep(ql::QLearnerWithExperienceReplay; env::AbstractMDP, kwargs...)
    ql.s = state(env)
    nothing
end

function poststep(ql::QLearnerWithExperienceReplay; env::AbstractMDP, rng::AbstractRNG, kwargs...)
    @unpack π, q, γ, counts, s, buffer, batch_size = ql

    a, r, s′, d = action(env), reward(env), state(env), in_absorbing_state(env)
    δ = r + (1 - Float64(d)) * γ * sum(a′ -> π(s′, a′) * q[a′, s′], action_space(env)) - q[a, s]
    counts[a, s] += 1
    α = 1.0 / counts[a, s]
    q[a, s] += α * δ
    
    push!(buffer, (s, a, r, s′, d))

    # while true
        # replay_counts = zero(counts)
        # q_ = copy(q)
        for i in 1:min(length(buffer), batch_size)
            (s, a, r, s′, d) = length(buffer) > batch_size ? rand(rng, buffer) : buffer[i]
            δ = r + (1 - Float64(d)) * γ * sum(a′ -> π(s′, a′) * q[a′, s′], action_space(env)) - q[a, s]
            # replay_counts[a, s] += 1
            # α = 1.0 / replay_counts[a, s]
            # α = 0.01
            α = 1/counts[a, s]
            q[a, s] += α * δ
        end
        # rel_change = maximum((abs.(q - q_)) ./ (abs.(q) .+ 1e-6))
        # rel_change = maximum((abs.(q - q_)))
        # println(length(buffer), " ", rel_change)
        # println(buffer)
        # println()
        # rel_change < 0.001 && break
    # end
    # println("broken")
    nothing
end
