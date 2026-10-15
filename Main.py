from Network_Archetecture import *
parameters = parameter_intitialisation()
n_epochs = int(input("Enter the number of epochs: "))
for epoch in range(n_epochs):
    #Front Propogation
    storage1 = front_propogation(X,parameters,one_hot_encoder(train_labels,N_Classes = 10))

    #Back Propogation
    gradients = back_propogation(one_hot_encoder(train_labels,N_Classes = 10),parameters,storage1)

    #weights and biases update
    parameters = update_parameters(parameters,gradients,0.1)
print("Training Complete")
print("Final Loss: "+f"{storage1["Loss"]:.2f}")
print("Final Accuracy: "+f"{np.mean(np.argmax(storage1["Zsoft"],axis=1) == train_labels)*100:.2f}%")


#Testing the model on the test dataset
result = Test_Model(test_images_n,parameters)
for i in range(test_labels.shape[0]):
    print(f"Image {i+1}: Predicted Class: {np.argmax(result[i])}, Actual Class: {test_labels[i]}")


