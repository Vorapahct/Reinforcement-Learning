using POMDPs

const GWPos = SVector{2,Int} #just make a vector of ints with size 2 (2 value tuple basically)

# @with_kw = Macro which allows default values for field types and a few other features.

@with_kw struct SimpleGridWorldIordan <: MDP{GWPos, Symbol}  #mdp that takes in GWPos object and symbol
    size::Tuple{Int, Int}           = (5,5)
    rewards::Dict{GWPos, Float64}   = Dict(GWPos(1,1)=>-10.0)#, GWPos(1,5)=>-5.0, GWPos(5,1)=>10.0, GWPos(5,5)=>3.0)  #dictionary of co-ordinates with reward amounts 					           
    terminate_from::Set{GWPos}      = Set(keys(rewards)) #creates a set of svector keys based off the dict. 
    tprob::Float64                  = 1.0 #0.7
    discount::Float64               = 0.95
end


# States

function POMDPs.states(mdp::SimpleGridWorldIordan)
    ss = vec(GWPos[GWPos(x, y) for x in 1:mdp.size[1], y in 1:mdp.size[2]]) #create a vector of all the co-ordiante pairs
    push!(ss, GWPos(-1,-1)) #add [-1,-1] to that list
    return ss
end

POMDPs.n_states(mdp::SimpleGridWorldIordan) = prod(mdp.size) + 1 #get the number of states including terminal/null state

function POMDPs.stateindex(mdp::SimpleGridWorldIordan, s::AbstractVector{Int})
    if all(s.>0) #element wise comparison of each element
        return LinearIndices(mdp.size)[s...] #get the index number for that key/co-ordinate
    else
        return n_states(mdp) 
    end
end

struct GWUniform
    size::Tuple{Int, Int}
end

Base.rand(rng::AbstractRNG, d::GWUniform) = GWPos(rand(rng, 1:d.size[1]), rand(rng, 1:d.size[2])) #pick 2 random numbers for a complete random co-orindate

function POMDPs.pdf(d::GWUniform, s::GWPos)   #returns a pdf of the state space. it is an equal distribution(0.04) for each state. override internal function. Evaluate the probability density of distribution d at sample s.
    if all(1 .<= s[1] .<= d.size) #check whether the diemnsions allow?
        return 1/prod(d.size)
    else
        return 0.0
    end
end

POMDPs.support(d::GWUniform) = (GWPos(x, y) for x in 1:d.size[1], y in 1:d.size[2]) #Return an iterable object containing the possible values that can be sampled from distribution d. override internal support function

POMDPs.initialstate_distribution(mdp::SimpleGridWorldIordan) = GWUniform(mdp.size)

# Actions

POMDPs.actions(mdp::SimpleGridWorldIordan) = (:up, :down, :left, :right) #just lists the actions
Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L = t[rand(rng, 1:length(t))] # don't know why this doesn't work out of the box
POMDPs.n_actions(mdp::SimpleGridWorldIordan) = 4 #just lists the number of actions

const dir = Dict(:up=>GWPos(0,1), :down=>GWPos(0,-1), :left=>GWPos(-1,0), :right=>GWPos(1,0)) #dictionary that maps direction symbols to y,x values
const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4) #action index dictionary

POMDPs.actionindex(mdp::SimpleGridWorldIordan, a::Symbol) = aind[a] #use the dictionary to get the action index


# Transitions

POMDPs.isterminal(m::SimpleGridWorldIordan, s::AbstractVector{Int}) = any(s.<0) #check if any value is less than 0, which means its terminal/state 26


# terminate_from is a dict of all the reward states, making reward states also terminal states
function POMDPs.transition(mdp::SimpleGridWorldIordan, s::AbstractVector{Int}, a::Symbol)
    if s in mdp.terminate_from || isterminal(mdp, s) #check if we are in a terminal state or not
        return Deterministic(GWPos(-1,-1))    #Create a deterministic distribution over only one value
    end

    destinations = MVector{n_actions(mdp)+1, GWPos}(undef) #create a mutable vector of undefined co-ordinates
    destinations[1] = s #set first element to the s provided

    # probs = MVector{n_actions(mdp)+1, Float64}()
    probs = @MVector(zeros(n_actions(mdp)+1))  #probability of moving????
    for (i, act) in enumerate(actions(mdp))
        if act == a
            prob = mdp.tprob # probability of transitioning to the desired cell
        else
            prob = (1.0 - mdp.tprob)/(n_actions(mdp) - 1) # probability of transitioning to another cell
        end

        dest = s + dir[act]
        destinations[i+1] = dest

        if !inbounds(mdp, dest) # hit an edge and come back
            probs[1] += prob
        else
            probs[i+1] += prob
        end
    end

    return SparseCat(destinations, probs) #returns the states with their probabilities
end

function inbounds(m::SimpleGridWorldIordan, s::AbstractVector{Int})
    return 1 <= s[1] <= m.size[1] && 1 <= s[2] <= m.size[2] #checks the correct ranges
end

# Rewards

POMDPs.reward(mdp::SimpleGridWorldIordan, s::AbstractVector{Int}) = get(mdp.rewards, s, 0.0)
POMDPs.reward(mdp::SimpleGridWorldIordan, s::AbstractVector{Int}, a::Symbol) = reward(mdp, s)

# discount

POMDPs.discount(mdp::SimpleGridWorldIordan) = mdp.discount

# Conversion
function POMDPs.convert_a(::Type{V}, a::Symbol, m::SimpleGridWorldIordan) where {V<:AbstractArray}
    convert(V, [aind[a]])
end

function POMDPs.convert_a(::Type{Symbol}, vec::V, m::SimpleGridWorldIordan) where {V<:AbstractArray}
    actions(m)[convert(Int, first(vec))]
end
