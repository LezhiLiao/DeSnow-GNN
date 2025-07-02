def registration(x,edge_index,registration_dis):#out就是xj

    x_i = x[:,0:3][edge_index[1]]
    x_j = x[:,0:3][edge_index[0]]
    x_r = registration_dis[edge_index[1]]
    result = x_j - x_i + x_r
    return result