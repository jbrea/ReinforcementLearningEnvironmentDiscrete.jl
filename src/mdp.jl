using Distributions
"""
    mutable struct MDP{T,R}
        observationspace::DiscreteSpace
        actionspace::DiscreteSpace
        state::Int64
        trans_probs::Array{T, 2}
        reward::R
        initialstates::Array{Int64, 1}
        isterminal::Array{Int64, 1}

A Markov Decision Process with `ns` states, `na` actions, current `state`,
`na`x`ns` - array of transition probabilites `trans_props` which consists for
every (action, state) pair of a (potentially sparse) array that sums to 1 (see
[`getprobvecrandom`](@ref), [`getprobvecuniform`](@ref),
[`getprobvecdeterministic`](@ref) for helpers to constract the transition
probabilities) `na`x`ns` - array of `reward`, array of initial states
`initialstates`, and `ns` - array of 0/1 indicating if a state is terminal.
"""
mutable struct MDP{T,R}
    observationspace::DiscreteSpace
    actionspace::DiscreteSpace
    state::Int64
    trans_probs::Array{T, 2}
    reward::R
    initialstates::Array{Int64, 1}
    isterminal::Array{Int64, 1}
end
function MDP(ospace, aspace, state, trans_probs::AbstractArray{T, 2},
             reward::R, initialstates, isterminal) where {T, R}
    if R <: AbstractMatrix
        reward = DeterministicStateActionReward(reward)
    end
    MDP{T,typeof(reward)}(ospace, aspace, state, trans_probs, reward, initialstates, isterminal)
end
function interact!(env::MDP, action)
    oldstate = env.state
    run!(env, action)
    r = reward(env.reward, oldstate, action, env.state)
    (observation = env.state, reward = r, isdone = env.isterminal[env.state] == 1)
end
function getstate(env::MDP)
    (observation = env.state, isdone = env.isterminal[env.state] == 1)
end
function reset!(env::MDP)
    env.state = rand(ENV_RNG, env.initialstates)
    (observation = env.state, )
end

mutable struct ChangeMDP{TMDP}
    ns::Int64
    actionspace::DiscreteSpace
    changeprobability::Float64
    stochasticity::Float64
    mdp::TMDP
    switchflag::Bool
    seed::Any
    rng::MersenneTwister # Used only for switches!
end
function ChangeMDP(; ns = 10, na = 4, nrewards = 2,
                    changeprobability = .01, stochasticity = 0.1,
                    seed = 3)
    rng = MersenneTwister(seed)
    mdpbase = MDP(ns, na, nrewards, init = "random")
    T = [rand(ENV_RNG, Dirichlet(ns, stochasticity)) for a in 1:na, s in 1:ns]
    mdpbase.trans_probs = deepcopy(T)
    ChangeMDP(ns, DiscreteSpace(na, 1), changeprobability, stochasticity,
                mdpbase, false, seed, rng)
end
export ChangeMDP
getstate(env::ChangeMDP) = getstate(env.mdp)
reset!(env::ChangeMDP) = reset!(env.mdp)
function interact!(env::ChangeMDP, action)
    env.switchflag = false
    r = rand(env.rng)
    # @show r
    if r < env.changeprobability # Switch or not!
        #println("Switch!")
        # Pick an s-a pair and change it: (Option #2)
        a = rand(env.rng, 1:env.mdp.actionspace.n)
        s = rand(env.rng, 1:env.ns)
        T = rand(env.rng, Dirichlet(env.ns, env.stochasticity))
        env.mdp.trans_probs[a, s] = deepcopy(T)
        # #Change currect s-a pair: (Option #1)
        # env.mdp.trans_probs[action, env.mdp.state] = deepcopy(T)
        # # Change the whole MDP: (Option #3)
        # T = [rand(env.rng, Dirichlet(env.ns, env.stochasticity))
        #         for a in 1:env.mdp.actionspace.n, s in 1:env.ns]
        # env.mdp.trans_probs = deepcopy(T)
        env.switchflag = true
    end
    interact!(env.mdp, action)
end
actionspace(env::ChangeMDP) = actionspace(env.mdp)
"""
    struct DeterministicNextStateReward
        value::Vector{Float64}
"""
struct DeterministicNextStateReward
    value::Vector{Float64}
end
reward(r::DeterministicNextStateReward, s, a, s′) = r.value[s′]
"""
    struct DeterministicStateActionReward
        value::Array{Float64, 2}

`value` should be a `na × ns`-matrix.
"""
struct DeterministicStateActionReward
    value::Array{Float64, 2}
end
reward(r::DeterministicStateActionReward, s, a, s′) = r.value[a, s]
"""
    struct NormalNextStateReward
        mean::Vector{Float64}
        std::Vector{Float64}
"""
struct NormalNextStateReward
    mean::Vector{Float64}
    std::Vector{Float64}
end
reward(r::NormalNextStateReward, s, a, s′) = r.mean[s′] + randn(ENV_RNG) * r.std[s′]
"""
    struct NormalStateActionReward
        mean::Array{Float64, 2}
        std::Array{Float64, 2}

`mean` and `std` should be `na × ns`-matrices.
"""
struct NormalStateActionReward
    mean::Array{Float64, 2}
    std::Array{Float64, 2}
end
reward(r::NormalStateActionReward, s, a, s′) = r.mean[a, s] + randn(ENV_RNG) * r.std[a, s]

"""
    getprobvecrandom(n)

Returns an array of length `n` that sums to 1. More precisely, the array is a
sample of a [Dirichlet
distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) with `n`
categories and ``α_1 = ⋯  = α_n = 1``.
"""
getprobvecrandom(n) = normalize(rand(ENV_RNG, n), 1)
"""
    getprobvecrandom(n, min, max)

Returns an array of length `n` that sums to 1 where all elements outside of
`min`:`max` are zero.
"""
getprobvecrandom(n, min, max) = sparsevec(collect(min:max),
                                          getprobvecrandom(max - min + 1), n)
"""
    getprobvecuniform(n)  = fill(1/n, n)
"""
getprobvecuniform(n) = fill(1/n, n)
"""
    getprobvecdeterministic(n, min = 1, max = n)

Returns a `SparseVector` of length `n` where one element in `min`:`max` has
value 1.
"""
getprobvecdeterministic(n, min = 1, max = n) = sparsevec([rand(ENV_RNG, min:max)], [1.], n)
# constructors
"""
    MDP(ns, na; init = "random")
    MDP(; ns = 10, na = 4, init = "random")

Return MDP with `init in ("random", "uniform", "deterministic")`, where the
keyword init determines how to construct the transition probabilites (see also
[`getprobvecrandom`](@ref), [`getprobvecuniform`](@ref),
[`getprobvecdeterministic`](@ref)).
"""
function MDP(ns, na; init = "random")
    r = randn(ENV_RNG, na, ns)
    func = eval(Meta.parse("getprobvec" * init))
    T = [func(ns) for a in 1:na, s in 1:ns]
    MDP(DiscreteSpace(ns, 1), DiscreteSpace(na, 1), rand(ENV_RNG, 1:ns), T, r,
        1:ns, zeros(ns))
end
MDP(; ns = 10, na = 4, init = "random") = MDP(ns, na, init = init)
function MDP(ns, na, r::Vector{Float64}; init = "random")
    func = eval(Meta.parse("getprobvec" * init))
    T = [func(ns) for a in 1:na, s in 1:ns]
    reward = DeterministicNextStateReward(r)
    MDP(DiscreteSpace(ns, 1), DiscreteSpace(na, 1), rand(ENV_RNG, 1:ns), T, reward,
        1:ns, zeros(ns))
end
function MDP(ns, na, nrewards::Int; init = "random")
    r = zeros(ns)
    [r[i] = 1. for i in rand(ENV_RNG, 1:ns, nrewards)]
    reward = DeterministicNextStateReward(r)
    isterminal = Int.(r)
    func = eval(Meta.parse("getprobvec" * init))
    T = [func(ns) for a in 1:na, s in 1:ns]
    MDP(DiscreteSpace(ns, 1), DiscreteSpace(na, 1), rand(ENV_RNG, 1:ns), T, reward,
        1:ns, isterminal)
end
actionspace(env::MDP) = env.actionspace

"""
    treeMDP(na, depth; init = "random", branchingfactor = 3)

Returns a tree structured MDP with na actions and `depth` of the tree.
If `init` is random, the `branchingfactor` determines how many possible states a
(action, state) pair has. If `init = "deterministic"` the `branchingfactor =
na`.
"""
function treeMDP(na, depth;
                 init = "random",
                 branchingfactor = 3)
    isdet = (init == "deterministic")
    if isdet
        branchingfactor = na
        ns = na.^(0:depth - 1)
    else
        ns = branchingfactor.^(0:depth - 1)
    end
    cns = cumsum(ns)
    func = eval(Meta.parse("getprobvec" * init))
    T = Array{SparseVector, 2}(undef, na, cns[end])
    for i in 1:depth - 1
        for s in 1:ns[i]
            for a in 1:na
                lb = cns[i] + (s - 1) * branchingfactor + (a - 1) * isdet + 1
                ub = isdet ? lb : lb + branchingfactor - 1
                T[a, (i == 1 ? 0 : cns[i-1]) + s] = func(cns[end] + 1, lb, ub)
            end
        end
    end
    r = zeros(na, cns[end] + 1)
    isterminal = [zeros(cns[end]); 1]
    for s in cns[end-1]+1:cns[end]
        for a in 1:na
            r[a, s] = -rand(ENV_RNG)
            T[a, s] = getprobvecdeterministic(cns[end] + 1, cns[end] + 1,
                                              cns[end] + 1)
        end
    end
    MDP(DiscreteSpace(cns[end] + 1, 1), DiscreteSpace(na, 1), 1, T, r, 1:1, isterminal)
end

function emptytransprob!(v::SparseVector)
    empty!(v.nzind); empty!(v.nzval)
end
emptytransprob!(v::Array{Float64, 1}) = v[:] .*= 0.

"""
    setterminalstates!(mdp, range)

Sets `mdp.isterminal[range] .= 1`, empties the table of transition probabilities
for terminal states and sets the reward for all actions in the terminal state to
the same value.
"""
function setterminalstates!(mdp, range)
    mdp.isterminal[range] .= 1
    for s in findall(mdp.isterminal)
        mdp.reward[:, s] .= mean(mdp.reward[:, s])
        for a in 1:mdp.na
            emptytransprob!(mdp.trans_probs[a, s])
        end
    end
end

# run MDP


"""
    run!(mdp::MDP, action::Int64)

Transition to a new state given `action`. Returns the new state.
"""
function run!(mdp::MDP, action::Int64)
    if mdp.isterminal[mdp.state] == 1
        reset!(mdp)
    else
        mdp.state = wsample(ENV_RNG, mdp.trans_probs[action, mdp.state])
        (observation = mdp.state,)
    end
end

"""
    run!(mdp::MDP, policy::Array{Int64, 1}) = run!(mdp, policy[mdp.state])

"""
run!(mdp::MDP, policy::Array{Int64, 1}) = run!(mdp, policy[mdp.state])
