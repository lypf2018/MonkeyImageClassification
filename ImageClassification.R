require(jpeg)
require(OpenImageR)
require(RCurl)
library(keras)

training_path <- "http://www.utdallas.edu/~yxs173830/cnnproject1/training/n"
validation_path <- "http://www.utdallas.edu/~yxs173830/cnnproject1/validation/n"
training_data <- list()
training_label <- list()
training_nums <- c(111, 110, 122, 105, 113, 106, 114, 106, 104, 105)
validation_data <- list()
validation_label <- list()
validation_nums <- c(28, 27, 30, 26, 28, 26, 28, 27, 26, 26)

index <- 0
for (i in 1:10) {
  for(j in 1:training_nums[i]){
    original_image <- readJPEG(getURLContent(paste(training_path,i,"/n",i,"%20(",j,").jpg", sep = ""), binary=TRUE))
    rezised_image <- resizeImage(original_image, width = 50, height = 50, method = 'bilinear')
    training_data[[index+j]] <- rezised_image
    training_label[index+j] <- i
  }
  index <- index + training_nums[i]
}

index <- 0
for (i in 1:10) {
  for(j in 1:validation_nums[i]){
    original_image <- readJPEG(getURLContent(paste(validation_path,i,"/n",i,"%20(",j,").jpg", sep = ""), binary=TRUE))
    validation_data[[index+j]] <- resizeImage(original_image, width = 50, height = 50, method = 'bilinear')
    validation_label[index+j] <- i
  }
  index <- index + validation_nums[i]
}

training_label <- array(as.numeric(unlist(training_label)))
validation_label <- array(as.numeric(unlist(validation_label)))
View(validation_label)

train_image_array<-array(dim = c(length(training_data),50,50,3))
for (i in 1:length(training_data)) {
  for (j in 1:50) {
    for (k in 1:50) {
      for (l in 1:3) {
        train_image_array[i,j,k,l]<-training_data[[i]][j,k,l]
      }
    }
  }
}

validation_image_array<-array(dim = c(length(validation_data),50,50,3))
for (i in 1:length(validation_data)) {
  for (j in 1:50) {
    for (k in 1:50) {
      for (l in 1:3) {
        validation_image_array[i,j,k,l]<-validation_data[[i]][j,k,l]
      }
    }
  }
}

# train_label <- array(as.numeric(unlist(train_label)))
training_label_encode <- to_categorical(training_label)
# training_label_encode <- training_label_encode[,2:11]
View(training_label_encode)
validation_label_encode <- to_categorical(validation_label)
# validation_label_encode <- validation_label_encode[,2:11]

#model

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(50, 50, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

model <- model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 11, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history<-model %>% fit(
  train_image_array, training_label_encode, 
  epochs = 20, batch_size=64
)

results <- model %>% evaluate(validation_image_array, validation_label_encode)
results

predict_classes(model,validation_image_array)

