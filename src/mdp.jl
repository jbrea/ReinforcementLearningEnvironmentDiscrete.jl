using Distributions
"""
    mutable struct MDP
        ns::Int64
        na::Int64
        state::Int64
        trans_probs::Array{AbstractArray, 2}
        reward::Array{Float64, 2}
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
mutable struct MDP{T}
    observationspace::DiscreteSpace
    actionspace::DiscreteSpace
    state::Int64
    trans_probs::Array{T, 2}
    reward::Array{Float64, 2}
    initialstates::Array{Int64, 1}
    isterminal::Array{Int64, 1}
end
function MDP(ospace, aspace, state, trans_probs::Array{T, 2},
             reward, initialstates, isterminal) where T
    MDP{T}(ospace, aspace, state, trans_probs, reward, initialstates, isterminal)
end
function interact!(env::MDP, action)
    r = env.reward[action, env.state]
    run!(env, action)
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
    stayprobability::Float64
    stochasticity::Float64
    mdp::TMDP
    switchflag::Bool
end
function ChangeMDP(; ns = 10, na = 4, stayprobability = .99, stochasticity = 0.1)
    mdpbase = MDP(ns, na, init = "random")
    T = [rand(ENV_RNG, Dirichlet(ns, stochasticity)) for a in 1:na, s in 1:ns]
    mdpbase.trans_probs = copy(T)
    ChangeMDP(ns, DiscreteSpace(na, 1), stayprobability, stochasticity, mdpbase, false)
end
export ChangeMDP
getstate(env::ChangeMDP) = getstate(env.mdp)
reset!(env::ChangeMDP) = reset!(env.mdp)
function interact!(env::ChangeMDP, action)
    env.switchflag = false
    # Switch or not!
    r=aand()
    if r > env.stayprobability
        #println("Switch!")
        T =rand(ENV_RNG, Dirichlet(env.ns, env.stochasticity))
        env.mdp.trans_probs[action, env.mdp.state] = copy(T)
        env.switchflag = true
    end
    interact!(env.mdp, action)
end
actionspace(env::ChangeMDP) = actionspace(env.mdp)
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
