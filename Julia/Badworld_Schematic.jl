using POMDPs
using StaticArrays
using Parameters
using Random

const GWPos = SVector{2,Int} 

#S,A,T,y,R
@with_kw struct Badworld <: MDP{GWPos, Symbol}  
    size::Tuple{Int, Int}           = (5,5)
    rewards::Dict{GWPos, Float64}   = Dict(GWPos(1,1)=>-10.0)
    terminate_from::Set{GWPos}      = Set(keys(rewards))  
    discount::Float64               = 0.95
end

function POMDPs.states(mdp::Badworld)
    ss = vec(GWPos[GWPos(x, y) for x in 1:mdp.size[1], y in 1:mdp.size[2]]) 
    push!(ss, GWPos(-1,-1)) 
    return state_space
end

POMDPs.n_states(mdp::Badworld) = (mdp.size[1] * mdp.size[2]) + 1 

function POMDPs.stateindex(mdp::Badworld, s::AbstractVector{Int})
    if s == (-1,-1)
        return n_states(mdp) 
    else
        return LinearIndices(mdp.size)[s...]
    end
end

struct GWUniform
    size::Tuple{Int, Int}
end

Base.rand(rng::AbstractRNG, d::GWUniform) = GWPos(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2])) 

function POMDPs.pdf(d::GWUniform, s::GWPos)   
    if all(1 .<= s[1] .<= d.size) 
        return 1/prod(d.size)
    else
        return 0.0
    end
end

POMDPs.support(d::GWUniform) = (GWPos(x, y) for x in 1:d.size[1], y in 1:d.size[2])
POMDPs.initialstate_distribution(mdp::Badworld) = GWUniform(mdp.size)

# Actions

POMDPs.actions(mdp::Badworld) = (:up, :down, :left, :right) 
Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L = t[rand(rng, 1:length(t))] 
POMDPs.n_actions(mdp::Badworld) = 4 
const dir = Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0)) 
const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4) 
POMDPs.actionindex(mdp::Badworld, a::Symbol) = aind[a] 

# Transitions

POMDPs.isterminal(m::Badworld, s::AbstractVector{Int}) = any(s.<0)

# terminate_from is a dict of all the reward states, making reward states also terminal states
function POMDPs.transition(mdp::Badworld, s::AbstractVector{Int}, a::Symbol)
    if s in mdp.terminate_from || isterminal(mdp, s) 
        return Deterministic(GWPos(-1,-1))   
    end

    destinations = MVector{n_actions(mdp)+1, GWPos}(undef) 
    destinations[1] = s 

    probs = @MVector(zeros(n_actions(mdp)+1))
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = 1
        else
            prob = 0
        end

        dest = s + dir[act]
        destinations[i+1] = dest

        if !(inbounds(mdp, dest1 <= s[1] <= m.size[1] && 1 <= s[2] <= m.size[2])) 
            probs[1] += prob
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(destinations, probs) 
end

# Rewards

POMDPs.reward(mdp::Badworld, s::AbstractVector{Int}) = get(mdp.rewards, s, 0.0)
POMDPs.reward(mdp::Badworld, s::AbstractVector{Int}, a::Symbol) = reward(mdp, s)

# discount

POMDPs.discount(mdp::Badworld) = mdp.discount

# Conversion
function POMDPs.convert_a(::Type{V}, a::Symbol, m::Badworld) where {V<:AbstractArray}
    convert(V, [aind[a]])
end

function POMDPs.convert_a(::Type{Symbol}, vec::V, m::Badworld) where {V<:AbstractArray}
    actions(m)[convert(Int, first(vec))]
end
