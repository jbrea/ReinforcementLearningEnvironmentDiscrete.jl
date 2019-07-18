# Cliff walking (Sutton & Barto)
function CliffWalking(; offset = 0.,
                        goalreward = 0.,
                        cliffreward = -100.,
                        stepreward = -1.)
    ns = 4*12; na = 4
    mdp = MDP(ns, na, init = "deterministic")
    mdp.isterminal[45] = 1
    mdp.initialstates = [1]
    mdp.state = 1
    for s in 1:ns
        for a in 1:na
            mdp.reward.value[a, s] = stepreward
            if a == 4 && s <= 4 || # left wall
               a == 1 && mod(s, 4) == 0 || # top wall
               a == 2 && s > 44 || # right wall
               a == 3 && s == 1 # bottom wall (only relevant for initial state)
                mdp.trans_probs[a, s] = SparseVector(ns, [s], [1.])
            elseif (mod(s, 4) == 2 && a == 3 && s != 2 && s != 46) ||
                   (s == 1 && a == 2) # cliff
                mdp.trans_probs[a, s] = SparseVector(ns, [1], [1.])
                mdp.reward.value[a, s] = cliffreward
            else
                if a == 1 # up
                    mdp.trans_probs[a, s] = SparseVector(ns, [s+1], [1.])
                elseif a == 2 # right
                    mdp.trans_probs[a, s] = SparseVector(ns, [s+4], [1.])
                elseif a == 3 # down
                    mdp.trans_probs[a, s] = SparseVector(ns, [s-1], [1.])
                else # left
                    mdp.trans_probs[a, s] = SparseVector(ns, [s-4], [1.])
                end
            end
        end
    end
    mdp.reward.value[3, 45] = goalreward
    mdp.reward.value .+= offset
    m = ones(Int64, 4, 12)
    m[5:4:41] .= 0
    DiscreteMaze(mdp, m, [45], [goalreward], collect(1:48), collect(1:48), 0.)
end
