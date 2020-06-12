using SparseArrays, LinearAlgebra
struct Maze
    dimx::Int
    dimy::Int
    walls::Symmetric{Bool,SparseMatrixCSC{Bool,Int}}
end
function emptymaze(dimx, dimy)
    Maze(dimx, dimy, Symmetric(sparse([], [], Bool[], dimx * dimy, dimx * dimy)))
end
iswall(maze, pos, dir) = hitborder(maze, pos, dir) || maze.walls[pos, nextpos(maze, pos, dir)]
function setwall!(maze, pos, dir)
    npos = nextpos(maze, pos, dir)
    maze.walls.data[sort([pos, npos])...] = true
end
function nextpos(maze, pos, dir)
    dir == :up && return pos - 1
    dir == :down && return pos + 1
    dir == :left && return pos - maze.dimy
    dir == :right && return pos + maze.dimy
end
function hitborder(maze, pos, dir)
    dir == :up && pos % maze.dimy == 1 && return true
    dir == :down && pos % maze.dimy == 0 && return true
    dir == :left && pos <= maze.dimy && return true
    dir == :right && pos > maze.dimy * (maze.dimx - 1) && return true
    return false
end

function orthogonal_directions(dir)
    dir ∈ (:up, :down) && return (:left, :right)
    return (:up, :down)
end

function is_wall_neighbour(maze, pos)
    for dir in (:up, :down, :left, :right)
        iswall(maze, pos, dir) && return true
    end
    for npos in (nextpos(maze, pos, :up), nextpos(maze, pos, :down))
        for dir in (:left, :right)
            iswall(maze, npos, dir) && return true
        end
    end
    return false
end
function is_wall_tangential(maze, pos, dir)
    for ortho_dir in orthogonal_directions(dir)
        iswall(maze, pos, ortho_dir) && return true
    end
    return false
end
is_wall_ahead(maze, pos, dir) = iswall(maze, pos, dir)

function addrandomwall!(maze; rng = Random.GLOBAL_RNG)
    potential_startpos = filter(x -> !is_wall_neighbour(maze, x), 1:maze.dimx * maze.dimy)
    if potential_startpos == []
        @warn("Cannot add a random wall.")
        return maze
    end
    pos = rand(rng, potential_startpos)
    direction = rand(rng, (:up, :down, :left, :right))
    while true
        setwall!(maze, pos, orthogonal_directions(direction)[2])
        pos = nextpos(maze, pos, direction)
        is_wall_tangential(maze, pos, direction) && break
        if is_wall_ahead(maze, pos, direction)
            setwall!(maze, pos, orthogonal_directions(direction)[2])
            break
        end
    end
    return maze
end

function n_effective(n, f, list)
    N = n === nothing ? div(length(list), Int(1/f)) : n
    min(N, length(list))
end
function breaksomewalls!(m; f = 1/50, n = nothing, rng = Random.GLOBAL_RNG)
    wall_idxs = findall(m.walls.data)
    pos = sample(rng, wall_idxs, n_effective(n, f, wall_idxs), replace = false)
    m.walls.data[pos] .= false
    dropzeros!(m.walls.data)
    m
end
# function addobstacles!(m; f = 1/100, n = nothing, rng = Random.GLOBAL_RNG)
#     nz = findall(x -> x == 1, reshape(m, :))
#     pos = sample(rng, nz, n_effective(n, f, nz), replace = false)
#     m[pos] .= 0
#     m
# end
setTandR!(d) = for s in 1:d.maze.dimx * d.maze.dimy setTandR!(d, s) end
function setTandR!(d, s)
    T = d.mdp.trans_probs
    R = d.mdp.reward
    goals = d.goals
    ns = d.mdp.observationspace.n
    maze = d.maze
    for (aind, a) in enumerate((:up, :down, :left, :right))
        npos = iswall(maze, s, a) ? s : nextpos(maze, s, a)
        if d.neighbourstateweight > 0 && !(npos in goals)
            positions = [npos]
            weights = [1.]
            for dir in (:up, :down, :left, :right)
                if !iswall(maze, npos, dir)
                    push!(positions, nextpos(maze, npos, dir))
                    push!(weights, d.neighbourstateweight)
                end
            end
            weights /= sum(weights)
            T[aind, s] = sparsevec(positions, weights, ns)
        else
            T[aind, s] = sparsevec([npos], [1.], ns)
            if s in goals
                idx_goals = findfirst(x -> x == s, goals)
                R[s] = d.goalrewards[idx_goals]
            end
        end
    end
end

"""
    struct DiscreteMaze
        mdp::MDP
        maze::Array{Int, 2}
        goals::Array{Int, 1}
        statefrommaze::Array{Int, 1}
        mazefromstate::Array{Int, 1}
"""
struct DiscreteMaze{T}
    mdp::T
    maze::Maze
    goals::Array{Int, 1}
    goalrewards::Array{Float64, 1}
    neighbourstateweight::Float64
end
"""
    DiscreteMaze(; nx = 40, ny = 40, nwalls = div(nx*ny, 10), ngoals = 1,
                   goalrewards = 1, stepcost = 0, stochastic = false,
                   neighbourstateweight = .05)

Returns a `DiscreteMaze` of width `nx` and height `ny` with `nwalls` walls and
`ngoals` goal locations with reward `goalreward` (a list of different rewards
for the different goal states or constant reward for all goals), cost of moving
`stepcost` (reward = -`stepcost`); if `stochastic = true` the actions lead with
a certain probability to a neighbouring state, where `neighbourstateweight`
controls this probability.
"""
function DiscreteMaze(; nx = 40, ny = 40, nwalls = div(nx*ny, 20),
                        rng = Random.GLOBAL_RNG, kwargs...)
    m = emptymaze(nx, ny)
    for _ in 1:nwalls
        addrandomwall!(m, rng = rng)
    end
    breaksomewalls!(m, rng = rng)
    DiscreteMaze(m; rng = rng, kwargs...)
end
function DiscreteMaze(maze; ngoals = 1,
                            goalrewards = 1.,
                            stepcost = 0,
                            stochastic = false,
                            neighbourstateweight = stochastic ? .05 : 0.,
                            rng = Random.GLOBAL_RNG)
    na = 4
    ns = maze.dimx * maze.dimy
    legalstates = 1:ns
    T = Array{SparseVector{Float64,Int}}(undef, na, ns)
    goals = sort(sample(rng, legalstates, ngoals, replace = false))
    R = -ones(na, ns) * stepcost
    isterminal = zeros(Int, ns); isterminal[goals] .= 1
    isinitial = setdiff(legalstates, goals)
    res = DiscreteMaze(MDP(DiscreteSpace(ns, 1),
                           DiscreteSpace(na, 1),
                           rand(rng, legalstates),
                           T, R,
                           isinitial,
                           isterminal),
                       maze,
                       goals,
                       typeof(goalrewards) <: Number ? fill(goalrewards, ngoals) :
                                                       goalrewards,
                       neighbourstateweight)
    setTandR!(res)
    res
end

interact!(env::DiscreteMaze, a) = interact!(env.mdp, a)
reset!(env::DiscreteMaze) = reset!(env.mdp)
getstate(env::DiscreteMaze) = getstate(env.mdp)
actionspace(env::DiscreteMaze) = actionspace(env.mdp)

mutable struct ChangeDiscreteMaze{DiscreteMaze}
    discretemaze::DiscreteMaze
    stepcounter::Int
    switchstep::Int
    switchflag::Bool # Used for RecordSwitches callback
    pos_switchto0::Int
    pos_switchto1::Int
end
function ChangeDiscreteMaze(; switchstep = 10^2, stochastic = false,
                            neighbourstateweight = stochastic ? .05 : 0.)
    dm = DiscreteMaze()
    stepcounter = 0
    pos_switchto0 = rand(ENV_RNG, 1:length(reshape(dm.maze, :)))
    pos_switchto1 = rand(ENV_RNG, 1:length(reshape(dm.maze, :)))
    ChangeDiscreteMaze(dm, stepcounter, switchstep, false, pos_switchto0, pos_switchto1)
end
function ChangeDiscreteMaze(maze; switchstep = 10^2, stochastic = false,
                            neighbourstateweight = stochastic ? .05 : 0.)

    dm = DiscreteMaze(maze, compressed = false, stochastic = stochastic,
                        neighbourstateweight = neighbourstateweight)
    stepcounter = 0
    pos_switchto0 = rand(ENV_RNG, 1:length(reshape(maze, :)))
    pos_switchto1 = rand(ENV_RNG, 1:length(reshape(maze, :)))
    ChangeDiscreteMaze(dm, stepcounter, switchstep, false, pos_switchto0, pos_switchto1)
end
function ChangeDiscreteMaze(maze, pos_switchto0, pos_switchto1; switchstep = 10^2,
                            stochastic = false,
                            neighbourstateweight = stochastic ? .05 : 0.)

    dm = DiscreteMaze(maze, compressed = false, stochastic = stochastic,
                        neighbourstateweight = neighbourstateweight)
    stepcounter = 0
    ChangeDiscreteMaze(dm, stepcounter, switchstep, false, pos_switchto0, pos_switchto1)
end

function interact!(env::ChangeDiscreteMaze, action)
    env.stepcounter += 1
    env.switchflag = false
    # Switch or not!
    if env.stepcounter == env.switchstep
        # println("Switch!")
        env.switchflag = true
        setupswitch!(env)
    end
    interact!(env.discretemaze.mdp, action)
end
function setupswitch!(env)
    env.discretemaze.maze[env.pos_switchto0] = 0
    env.discretemaze.maze[env.pos_switchto1] = 1
    setTandR!(env.discretemaze)
end
# reset!(env::ChangeDiscreteMaze) = reset!(env.discretemaze.mdp)
# getstate(env::ChangeDiscreteMaze) = getstate(env.discretemaze.mdp)
# actionspace(env::ChangeDiscreteMaze) = actionspace(env.discretemaze.mdp)
# plotenv(env::ChangeDiscreteMaze) = plotenv(env.discretemaze)

mutable struct RandomChangeDiscreteMaze{DiscreteMaze}
    discretemaze::DiscreteMaze
    n::Int
    changeprobability::Float64
    switchflag::Bool # Used for RecordSwitches callback
end
function RandomChangeDiscreteMaze(; nx = 20, ny = 20, ngoals = 4, nwalls = 10,
                   compressed = false, stochastic = false,
                   neighbourstateweight = stochastic ? .05 : 0., n = 5,
                   changeprobability = 0.999)

    dm = DiscreteMaze(nx = nx, ny = ny, ngoals = ngoals, nwalls = nwalls,
                       compressed = compressed, stochastic = stochastic,
                       neighbourstateweight = neighbourstateweight)
    RandomChangeDiscreteMaze(dm, n, changeprobability, false)
end
function RandomChangeDiscreteMaze(maze; n = 5, changeprobability = 0.999)
    dm = DiscreteMaze(maze, compressed = false)
    RandomChangeDiscreteMaze(dm, n, changeprobability, false)
end

function interact!(env::RandomChangeDiscreteMaze, action)
    env.switchflag = false
    # Switch or not!
    r=rand(ENV_RNG)
    if r > env.changeprobability
        # println("Switch!")
        env.switchflag = true
        breaksomewalls!(env.discretemaze.maze, n = env.n)
        addobstacles!(env.discretemaze.maze, n = env.n)
        setTandR!(env.discretemaze)
    end
    interact!(env.discretemaze.mdp, action)
end
reset!(env::Union{ChangeDiscreteMaze,RandomChangeDiscreteMaze}) = reset!(env.discretemaze.mdp)
getstate(env::Union{ChangeDiscreteMaze,RandomChangeDiscreteMaze}) = getstate(env.discretemaze.mdp)
actionspace(env::Union{ChangeDiscreteMaze,RandomChangeDiscreteMaze}) = actionspace(env.discretemaze.mdp)
plotenv(env::Union{ChangeDiscreteMaze,RandomChangeDiscreteMaze}) = plotenv(env.discretemaze)

function plotenv(env::DiscreteMaze)
    goals = env.goals
    px = 6
    m = zeros(px*env.maze.dimx, px*env.maze.dimy)
    for i in 1:env.maze.dimx
        for j in 1:env.maze.dimy
            s = (i - 1) * env.maze.dimy + j
            iswall(env.maze, s, :down) && (m[(i-1)*px+1:i*px, j*px] .= -3)
            iswall(env.maze, s, :up) && (m[(i-1)*px+1:i*px, (j-1)*px+1] .= -3)
            iswall(env.maze, s, :right) && (m[i*px, (j-1)*px+1:j*px] .= -3)
            iswall(env.maze, s, :left) && (m[(i-1)*px+1, (j-1)*px+1:j*px] .= -3)
            if s == env.mdp.state
                val = -1
            elseif s in goals
                val = 1
            else
                continue
            end
            m[(i-1)*px+2:i*px-1, (j-1)*px+2:j*px-1] .= val
        end
    end
    imshow(m, clim = (-3, 3), colormap = 42)
end
