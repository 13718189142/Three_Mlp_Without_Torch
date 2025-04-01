# Three_Mlp_Without_Torch
文件介绍:
  1.	Layer.py：作为构建三层神经网络分类器的基石，Layer.py 文件提供了核心功能支持。文件内的Dense类，负责全连接层权重和偏置的初始化工作，通过前向传播计算该层输出，借助反向传播完成梯度计算，并且引入 L2 正则化策略，有效防止模型过拟合。而softmax函数，则将模型的原始输出转化为概率分布，广泛应用于多分类预测场景，为图像分类任务的实现提供关键支持。
  2.	activationfun.py：该文件定义了一系列激活函数类，包括 ReLU、LeakyReLU、tangenthyperbolic 以及 sigmoid 等。每个激活函数类均实现了前向和反向传播方法，前者将输入数据按特定规则进行转换，后者则在反向传播过程中计算梯度，确保模型训练的顺利进行。
  3.	model.py：文件中定义的ThreeLayerNet类，专门用于搭建三层神经网络。初始化过程中，使用者可灵活指定各层的神经元数量、L2 正则化参数，以及各层所使用的激活函数。ThreeLayerNet类不仅支持前向传播计算预测概率、反向传播求解梯度，还将梯度更新操作封装在update函数中，极大提升了代码的复用性和可维护性。此外，借助save()和load()函数，实现了模型参数的保存与加载，方便模型的持续优化和部署。
  4.	Train 文件：这是神经网络的训练文件，采用分批次训练的策略，每次训练时，依次进行前向传播、反向传播，并更新模型参数。训练结束后，输出训练好的模型，并记录训练过程中的误差数据，为模型性能评估和调优提供依据。
  5.	Parameter_search：此文件负责搜索参数的设置工作，涵盖隐藏层数、学习率、正则化参数等关键超参数，通过对这些参数的调整，帮助使用者找到最优的模型配置，提升模型的性能表现 
  6.  Model 文件夹存储实验保存得到的数据, 命名格式为网络的参数设置_model 如 h1_100h2_50actfun_relu_leakyreluL2_0_model, 隐藏层数100,50, 使用激活函数relu,leakyrelu,正则化参数L2=0
如何运行:
    直接运行Main 文件即可. (选择其中的内容注释掉即可, 如不需要测试直接注释掉即可)
    Main中调用Train函数返回训练后的模型及训练记录.  梯度下降法放置在Train中,运行示例代码:
      model = ThreeLayerNet(input_size, 512,256, output_size,L2=0,actfunlist=['relu','leakyrelu'])
      model,train_losses, val_losses, val_accuracies = train(model,X_train, y_train, X_val, y_val,learning_rate=1e-4,num_epochs=10)
    Main中调用Test 函数即可   示例代码:
      test_model = ThreeLayerNet(input_size, 512,256, output_size,L2=0,actfunlist=['relu','leakyrelu'])
      
      test_model.load(model_dir)
      test_acc = test(test_model, X_test, y_test)
    参数搜索, 示例代码:
    best_model = parameter_search(X_train, y_train, X_val, y_val, X_test, y_test,actfunlist=['relu','leakyrelu'],epoch=10)
    如需要修改搜索范围在parameter_search 中进行搜索即可
如何导入保存
    保存部分, 只需要在函数训练好后, 调用save函数即可, model.save(), 命名格式如文件介绍6中提到.
    导入部分, 只需要调用model.load(model_dir)函数即可, 其中model_dir 为保存参数的文件夹, 自动寻找匹配模型参数设置的路径, 如果未找到合适的网络参数则报告错误.
            如果特定想调用某个具体的文件, 不使用匹配机制的话, 使用load2函数, 示例代码:model.load2(r"D:\homework\MLZL\Model\h1_512h2_256actfun_relu_leakyreluL2_0_model.pkl")
Plot文件为绘图文件, 为ipynb格式逐个code块运行即可, 包含训练流程
