for i in range(3):
    plt.figure(figure_lst[i])
    plt.subplot(1,2,1)
    plt.title(figure_lst[i]+'_cur_policy')
    plt.xlabel('time(year)')
    plt.ylabel(y_lable_lst[i])
    for j in range(4):
        plt.plot(x_lst,data_lst[i][0,:,j],label='farmer_{}'.format(j+1))
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title(figure_lst[i] + '_best_policy')
    plt.xlabel('time(year)')
    plt.ylabel(y_lable_lst[i])
    for j in range(4):
        plt.plot(x_lst, data_lst[i][1,:, j], label='farmer_{}'.format(j + 1))
    plt.legend()