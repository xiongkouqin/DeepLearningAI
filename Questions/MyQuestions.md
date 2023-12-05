# Questions I met but still don't know how to solve

## Machine-Learning-Specialization

### 1. When cost function converge?

![Screenshot 2023-12-04 at 15.49.40](assets/Screenshot%202023-12-04%20at%2015.49.40.png)

![Screenshot 2023-12-04 at 15.49.34](assets/Screenshot%202023-12-04%20at%2015.49.34.png)

> 我感觉就是说这个选定的learning rate其实没有很大，对于整体来说是可以收敛的，但是呢这里的w[0]，对应的feature是size,它的取值范围相对于其他的feature来说大很多，导致这个偏导比其他的大，所以每次更新修改的比较多。所以这也表明这个feature可能需要scaling。