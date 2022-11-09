using MDPs
import MDPs: preexperiment, preepisode, prestep, poststep, postepisode, postexperiment
using UnPack

export QLearner

"""Learns Q-Values for a given policy using the expected SARSA TD rule"""
mutable struct QLearner <: AbstractHook
    π::AbstractPolicy{Int, Int}
    q::Matrix{Float64}
    counts::Matrix{Int}
    s::Int
    QLearner(π::AbstractPolicy{Int, Int}, q::Matrix{Float64}) = new(π, q, zeros(Int, size(q)), 1)
end

function prestep(ql::QLearner; env::AbstractMDP, kwargs...)
    ql.s = state(env)
end

function poststep(ql::QLearner; env::AbstractMDP, kwargs...)
    @unpack π, q, counts, s = ql
    a, r, s′, d, γ = action(env), reward(env), state(env), in_absorbing_state(env), discount_factor(env)
    δ = r + (1 - Float64(d)) * γ * sum(a′ -> π(s′, a′) * q[a′, s′], action_space(env)) - q[a, s]
    counts[a, s] += 1
    α = 1.0 / counts[a, s]
    q[a, s] += α * δ
end
