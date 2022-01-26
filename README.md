<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1200px-Jupyter_logo.svg.png" width="60" height="50">

# Panda-vs-Cat-and-Dog-Classifier
This project deals with the Kaggle dataset that contains pictures of pandas, cats and dogs. The goal of this project is to classify dogs and pandas from dog vs. pandas dataset and cats and pandas from cats vs. pandas dataset.

## Introduction
The features of these pictures are pixel values. There are 1000 samples for each category of the animal.  The project has been divided into two segments:
1.	**Convolutional Neural Network Model (CNN) 1** [(pandas & dogs images together), target =  0 (Dog) or 1(Panda)] : The neural network for solving this problem will have:
<ul>
  <li> (150,150,3) inputs in the first layer</li>
  <li> 1 output processing element in the output layer, since the result will be a binary (0 or 1)</li>
</ul>

2. **Convolutional Neural Network Model (CNN) 1** [(pandas & cats images together), target =  0 (Cat) or 1(Panda)]:The neural network for solving this problem will have:
<ul>
  <li>(150,150,3) inputs in the first layer </li>
  <li>1 output processing element in the output layer, since the result will be a binary (0 or 1) </li>
 </ul>

### [PART I] : Reorganizing the data <br>
The image files are copied in the following structure:

![image](https://user-images.githubusercontent.com/98129458/151079005-3da6a18b-a0fd-480d-9d35-4b469f850bfc.png)

Then these steps need to be followed:
<ol type= '(1)'>
  <li>Path to the directory where the original dataset was uncompressed </li>
  <li>Directory where we will store our smaller dataset </li>
  <li>Utility function to copy cat (respectively dog) images from index start_index to index end_index </li>
to the subdirectory new_base_dir/{subset_name}/cat (respectively dog). "subset_name" will be either "train", "validation", or "test". </li>
  <li>Create the training subset with the first 1000 images of each category. </li>
  <li>Create the validation subset with the next 500 images of each category. </li>
  <li>Create the test subset with the next 1000 images of each category. </li>
  </ol>
  
 ### [Part II]: The Panda-vs-Dog Classifier

For the classifier, 1600 images are kept for training and remaining 400 image are for validation.

1.	A simple model (pvdm1) is developed to classify Panda (1) vs. Dog (0)
<ul>
  <li>The first layer consists of a CONV2D with 32 filter with 3&#215;3 shape with the input as (150,150,3). The activation function ‘relu’ is used </li>
  <li>The layer is followed by MaxPooling2D with a shape of 2&#215;2 ,</li>
  <li>The next hidden layer consists of CONV2D (64 filters with shape 3&#215;3)and MaxPooling2D (shape 22) </li>
  <li>The next hidden layer consists of CONV2D (128 filters with shape 3&#215;3)and MaxPooling2D (shape 22) </li>
  <li>Then Flatten layer is added to flatten the output of the previous layer </li>
  <li>The next hidden layer consists of Dense layer with 512 processing elements with ‘relu’ activation function </li>
  <li>The output layer contains one processing element with ‘sigmoid’ activation function for binary output (1 or 0). </li>
  <li>The Optimizer is selected as ‘RMSprop’ with learning rate 1&#215;10-4. </li>
  <li>The loss function is selected as ‘binary_crossentropy’.</li>
  <li>For training and validation generator, batch size is considered as 20 for both. </li>
  </ul>
  
  ![image](https://user-images.githubusercontent.com/98129458/151080075-632b8e49-9031-47d0-b6c1-ca103d262a4e.png)
  
  <br>Fig. 1: Summary of pvdm1
  
After running the fit method for the above CNN model with training and validation data (steps_per_epoch = 80,epochs=30, validation_steps=20), combine plot of training and validation accuracy and losses per epoch are analyzed. Fig. 2  and 3 shows the plot of the training and validation accuracy and loss, respectively.
  
  ![image](https://user-images.githubusercontent.com/98129458/151080331-5f913b0c-47de-4312-b868-08130d443bdc.png)
  
  <br>Fig. 2: Accuracy of Panda vs. Dog Classifier 1 (pvdm1) Plot
  
  ![image](https://user-images.githubusercontent.com/98129458/151080469-fe5346b8-05df-4a13-91a4-aa95f4bc8a4c.png)
  
  <br> Fig. 3: Training and Validation Loss of Panda vs. Dog Classifier 1 (pvdm1) Plot
  
The last epoch parameter performance: <br>
loss: 0.0208 - acc: 0.9913 - val_loss: 0.4990 - val_acc: 0.9075
<br><br>
Here the model seems to converge in a fixed loss after 8 epochs for validation samples. Although the training loss decreases with the increment of the epoch, it would be ideal to stop after 8 epochs as the validation loss starts to increase and validation accuracy starts to decrease after that epoch. Fig 4 and Fig. 5 show the final training and validation accuracy and loss respectively.

![image](https://user-images.githubusercontent.com/98129458/151080734-c5196ec0-5b63-4e39-bbb4-cad5b4ba4515.png)

<br>Fig. 4: Accuracy of final Panda vs. Dog Classifier 1 (final pvdm1) Plot

![image](https://user-images.githubusercontent.com/98129458/151080776-0ac2f1e1-fbe2-4c8d-9f9d-6596ac7ff666.png)

<br>Fig. 5: Accuracy of final Panda vs. Dog Classifier 1 (final pvdm1) Plot

<br>
The last epoch performance parameters: <br>
loss: 0.1105 - acc: 0.9615 - val_loss: 0.3381 - val_acc: 0.8750 <br><br>

2. An improved model (pvdm2) has been designed for classifying Panda (1) vs. Dog (0). The dropout layer is introduced in this model to improve performance.

![image](https://user-images.githubusercontent.com/98129458/151080914-db8d1849-9031-473d-af0f-0bf13303ca1b.png)

<br>Fig. 6: Summary of pvdm2

<br><br> After running the fit method for the above CNN model with training and validation data (steps_per_epoch = 80,epochs=30, validation_steps=20), combine plot of training and validation accuracy and losses per epoch are analyzed. Fig. 7 and 8 show the plot of the training and validation accuracy and loss, respectively.
<br>

![image](https://user-images.githubusercontent.com/98129458/151081001-8986d869-bfd9-48b6-a499-d7170af04226.png)

<br>Fig. 7: Accuracy of Panda vs. Dog Classifier 2 (pvdm2) Plot

<br>

![image](https://user-images.githubusercontent.com/98129458/151081047-5f27b92e-2db0-41fd-be81-2c505da16432.png)

<br>Fig. 8: Training and validation loss of pvdm2 model

<br><br>The last epoch parameter performance: <br>
loss: 0.0438 - acc: 0.9858 - val_loss: 0.2578 - val_acc: 0.9275
<br><br>
Here the model seems to overfit after 19 epochs for validation samples. Although the training loss decreases with the increment of the epoch, it would be ideal to stop after 19 epochs as the validation loss starts to increase. However, validation accuracy starts to increase after that epoch. Fig 9 and Fig. 10 show the final training and validation accuracy and loss respectively.

![image](https://user-images.githubusercontent.com/98129458/151081299-655059a1-9647-43e5-9a2e-e11ab421c979.png)

<br>Fig. 9: Accuracy of final pvdm2 Plot
<br>

![image](https://user-images.githubusercontent.com/98129458/151081361-2969af65-67a9-4271-9a38-de4291616940.png)

<br>Fig. 10: Training and Validation of final pvdm2 Plot

<br>The last epoch performance parameters: <br>
loss: 0.0390 - acc: 0.9856 - val_loss: 0.2750 - val_acc: 0.9375

3. Third CNN model (pvdm3) is designed using “Data Augmentation” and batch size is 50

![image](https://user-images.githubusercontent.com/98129458/151081782-88344a59-91ed-427a-a725-7ddc69023ff3.png)

<br>Fig. 11: Summary of pvdm3
<br><br>
After running the fit method for the above CNN model with training and validation data (steps_per_epoch = 32,epochs=100, validation_steps=8), combine plot of training and validation accuracy and losses per epoch are analyzed. Fig. 12 and 13 show the plot of the training and validation accuracy and loss, respectively.
<br>

![image](https://user-images.githubusercontent.com/98129458/151081887-f5090023-1a87-47ec-88a1-e5c1a5c9dd89.png)

<br>Fig. 12: Training and Validation Accuracy of pvdm3 model

![image](https://user-images.githubusercontent.com/98129458/151082034-b48e47cd-a991-4baa-9d1a-87854501b047.png)

<br>Fig. 13: Training and Validation Loss of pvdm3 model

<br>The last epoch parameter performance:
loss: 0.1734 - acc: 0.9310 - val_loss: 0.2960 - val_acc: 0.9000 <br>
Here the model seems to overfit after 54 epochs for validation samples. Although the training loss decreases with the increment of the epoch, it would be ideal to stop after 54 epochs as the validation loss starts to increase after that epoch.  However, validation accuracy is increasing randomly as the epoch is incremented. Fig 14 and Fig. 15 show the final training and validation accuracy and loss respectively.

![image](https://user-images.githubusercontent.com/98129458/151082184-18bde240-71e6-4a52-94fb-8545191ba400.png)

<br>Fig. 14: Training and Validation Accuracy of final pvdm3 model

![image](https://user-images.githubusercontent.com/98129458/151082284-689686e2-1690-4aeb-8eca-361d5180be14.png)

<br> Fig. 15: Training and Validation Loss of final pvdm3 model

<br><br>The last epoch performance parameters: <br>
loss: 0.1050 - acc: 0.9600 - val_loss: 0.1824 - val_acc: 0.9525 <br><br>

### [Part III]: The Panda -vs- Cat Classifier

For the classifier, 1600 images are kept for training and remaining 400 image are for validation.

![image](https://user-images.githubusercontent.com/98129458/151082536-91537cbb-1877-44cf-b2cc-7c159008f96f.png)

<br>Fig. 16: Summary of pvcm1

After running the fit method for the above CNN model with training and validation data (steps_per_epoch = 80,epochs=30, validation_steps=20), combine plot of training and validation accuracy and losses per epoch are analyzed. Fig. 17 and 18 show the plot of the training and validation accuracy and loss respectively.

![image](https://user-images.githubusercontent.com/98129458/151082620-c0ccefe6-1e08-4f5f-9aca-f9d3307675ea.png)

<br>Fig. 17: Training and Validation Accuracy of pvcm1 model 

![image](https://user-images.githubusercontent.com/98129458/151082678-ed5f95cb-76a2-4acc-b47d-ac87404ddbc2.png)

<br>Fig. 18: Training and Validation Loss of pvcm1 model

<br><br>The last epoch performance parameter:
loss: 0.0120 - acc: 0.9949 - val_loss: 0.2907 - val_acc: 0.9400
<br><br>
From Fig. 17, the model seems to overfit after 8 epochs for validation samples. Although the training loss decreases with the increment of the epoch, it would be logical to stop after 8 epochs as the validation loss starts to increase and validation accuracy starts to decrease after that epoch. Fig 18 and Fig. 19 show the final training and validation accuracy and loss respectively.

![image](https://user-images.githubusercontent.com/98129458/151082804-ed4dabb7-e018-417c-8639-d7509964a4e9.png)

<br>Fig. 19: Training and Validation Accuracy of Final pvcm1 Model 

![image](https://user-images.githubusercontent.com/98129458/151082842-d4dd750c-dbdc-4422-a03c-49a10bb50c36.png)

<br>Fig. 20: Training and Validation Loss of Final pvcm1 Model 
<br><br>The last epoch performance parameters: <br>
loss: 0.1573 - acc: 0.9460 - val_loss: 0.1797 - val_acc: 0.9450 <br><br>

2. A further improved CNN model (pvcm2) is developed that uses dropout layer. 

![image](https://user-images.githubusercontent.com/98129458/151083000-d9fcea69-0f23-4d77-a044-c8b187629027.png)

<br>Fig. 21: Summary of pvcm2 model 

<br><br>After running the fit method for the above CNN model with training and validation data (steps_per_epoch = 80, epochs=30, validation_steps=20), combine plot of training and validation accuracy and losses per epoch are analyzed. Fig. 22 and 23 show the plot of the training and validation accuracy and loss respectively.

![image](https://user-images.githubusercontent.com/98129458/151084233-77630111-13ad-42a5-ae54-8538e9d7ecc1.png)

<br>Fig. 22: Training and Validation Accuracy of pvcm2 Model 

![image](https://user-images.githubusercontent.com/98129458/151084314-1c22c876-e079-4ee2-a8bb-dc0fd442b54a.png)

<br>Fig. 23: Training and Validation Loss of pvcm2 Model 

<br><br>At 13th epoch: loss: 0.1608 - acc: 0.9281 - val_loss: 0.1883 - val_acc: 0.9350
<br>
The last epoch performance parameter: <br>
loss: 0.0468 - acc: 0.9834 - val_loss: 0.1960 - val_acc: 0.9325

<br><br>From Fig. 23, the model seems to overfit after 13 epochs for validation samples. Although the training loss decreases with the increment of the epoch, it would be advantageous to stop after 13 epochs as the validation loss starts to increase after that epoch. However, validation accuracy is increasing and decreases randomly for the increment. Fig 24 and Fig. 25 show the final training and validation accuracy and loss respectively.

![image](https://user-images.githubusercontent.com/98129458/151084427-8be4dc66-af34-4d76-9d42-1d05ef729104.png)

<br> Fig. 24: Training and Validation Accuracy of Final pvcm2 Model 

![image](https://user-images.githubusercontent.com/98129458/151084482-4475dda8-38b3-40e6-8d2c-06d01226888a.png)

<br>Fig. 25: Training and Validation Loss of Final pvcm2 Model 

<br><br>The last epoch performance parameters: <br>
loss: 0.1487 - acc: 0.9442 - val_loss: 0.1914 - val_acc: 0.9325 <br><br>

3. A third CNN model (pdcm3) using the data augmentation is developed in this section. 

![image](https://user-images.githubusercontent.com/98129458/151084613-484aa193-9478-462a-820b-2c9aacfdc9f8.png)

<br> Fig.26: Summary of pvcm3 Model

<br><br>After running the fit method for the above CNN model with training and validation data (steps_per_epoch = 32,epochs=100, validation_steps=8), combine plot of training and validation accuracy and losses per epoch are analyzed. Fig. 27 and 28 show the plot of the training and validation accuracy and loss respectively.

![image](https://user-images.githubusercontent.com/98129458/151084694-23133fe8-6584-49a7-91bb-3868830fde36.png)

<br>Fig. 27: Training and Validation Accuracy of pvcm3 Model 

![image](https://user-images.githubusercontent.com/98129458/151084732-5fac50ae-cbdf-4e0d-8107-ac870e905ff5.png)

<br>Fig. 28: Training and Validation Loss of pvcm3 Model 

<br><br>The last epoch performance parameter: <br>
loss: 0.1206 - acc: 0.9608 - val_loss: 0.2660 - val_acc: 0.9050 <br><br>

From Fig. 28, the model seems to overfit after 60 epochs for validation samples. Although the training loss decreases with the increment of the epoch, it would be advantageous to stop after 60 epochs as the validation loss starts to increase after that epoch. However, validation accuracy is increasing randomly (some of them decreased) for the increment. Fig 29 and Fig. 30 show the final training and validation accuracy and loss respectively.

![image](https://user-images.githubusercontent.com/98129458/151084927-b5a209b8-89ec-41b9-a6b4-3efeeff9d6ac.png)

<br> Fig. 29: Training and Validation Accuracy of Final pvcm3 Model 

![image](https://user-images.githubusercontent.com/98129458/151084964-16949212-105d-4983-b1fe-eb3bc1dcace2.png)

<br> Fig. 30: Training and Validation Loss of Final pvcm3 Model 

<br><br>The last epoch performance parameters: <br>
loss: 0.1911 - acc: 0.9222 - val_loss: 0.1971 - val_acc: 0.9225

### Conclusion
The panda vs. dog classifier has three model for this project. The first model (pvdm1) is a simple model which yields good performance on training data and very poor performance in validation data. The training loss for both pvdm1 and final_pvdm1 are low as expected.  The validation loss is 0.4990 and validation accuracy is 90.75% for the initial pvdm1. The final model is stopped at advantageous epoch(=8). Although the final model provides less validation loss (0.3381), the validation accuracy decreased to 87.50%. The second model (pvdm2) is improved with dropout layer. The training loss for the initial pvdm2 is 0.0438 and final pvdm2 has training loss of 0.0390. Both model on the second section are able to decrease the validation loss from the previous model. The initial pvdm2 increases the validation accuracy to 92.75%. Whereas final pvdm2 yields validation loss of 0.2750 and the validation accuracy of 93.75% at advantageous epoch of 19. The validation loss is higher than expected. Lastly, the pvdm3 model with data augmentation yields training loss and accuracy as 0.1734 and 0.9310, respectively. The performance for training is expected. However, the validation loss is quite high (0.2960). Moreover, the validation accuracy is also decreased. The final pvdm3 model is developed by stopping the model at the epoch of 54. As a result, the model comparatively gives lower validation loss (=0.1824) and higher validation accuracy (95.25%).  <br><br>

The panda vs. cat classifier has three models. The first model (pvcm1) provides very good performance on the training data as expected. However, the performance is degraded in the validation data as expected (val_loss = 0.2907). The final pvcm1 model is stopped at advantageous epoch (=8). As a result, the validation loss is decreased (0.1797) and the validation accuracy is increased as expected. The second model (pvcm2) is improved by the dropout layer. The validation loss (0.1960) is greatly reduced in this model from the previous model. The training accuracy has expected performance. The final model for pvcm2 is trained until 13 epochs. The final model yields a slightly validation loss (0.1914). The validation accuracy for this final model stays the same (0.9325). The third model (pvcm3) is developed using data augmentation. Initially, it yields great training accuracy and loss performance as expected. The validation loss is found as 0.2660 and validation accuracy is at 90.50%. The advantageous epoch for final pvdcm3 model is determined as 60. With this epoch number, the final model yields validation loss of 0.1971 and validation accuracy of 0.9225. However, both the training loss and accuracy are deteriorated.

<br><br>Both classifiers show that, the dropout procedure and data augmentation help to reduce the validation loss with tolerable decrease in validation accuracy. The further improvement can be done using more training data as well as additional pre-processing of the data. Mainly, these two projects help me learn about measurement of the performance of a network and the tuning performance hyperparameters to reduce overfitting.

























