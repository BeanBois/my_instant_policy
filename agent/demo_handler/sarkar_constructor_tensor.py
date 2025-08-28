


def cluster_demos_on_angular_granulity(demo_observations, demo_actions,  angular_granulity):
    # take initial orientation to be 0 
    initial_orientation = 0 # we work in deg for easier handling 
    # assuming that actions are accumulated we cluster them based on +-angular_granulity
    clusters = {}
    curr_parent = demo_observations[0]['time']
    childrens = []
    for i in range(1,len(demo_observations)):
        action = demo_actions[i]
        deg_turn = action[2]
        if abs(deg_turn) > angular_granulity:
            clusters[curr_parent] = childrens
            curr_parent = demo_observations[i]['time']
            childrens = []
            continue
        time_index = demo_observations[i]['time']
        childrens.append(time_index)
    return clusters

def cluster_demo(demo_agent_info, # [B, N, L, A, 6] x,y,theta,state,time,done
                 angular_granulities = [60,30,15]):
    # basically this function makes like a topological tree as such 
    # after clustering from theta1 1 : (2,3,4,5,6) & 7 : (8,9,10,11,12)
    # after clustering from theta2 2 : (3) & 4 : (5,6) &  8: (9) & 10 : (11,12)

    # clustering is done based on 2 conditions
        # if next proceeding agent_info state has changed, then the next_proceeding obs becomes new parent
        # else if abs(angle_parent - angle_next) > angular_granulity, then it becomes new parent
    # the desired tree is as such 
    #                   empty root 
    #                  /         \
    #                 1           7
    #                / \         / \
    #               2   4        8  10
    #              /.  /  \.     |.  | \
    #             3.  5.   6.    9.  11 12

    # then from this tree i want to encode use SK constructor to construct PC embeddings of hyp_dim
    # you can do this in pytorch




    
