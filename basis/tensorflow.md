# Tensorflow

## Basic process

- build the FLAGS

- deal the error

- load the train/test data

- build the graph: 
  $\rightarrow​$definite the tensor/variable/placeholder
  $\rightarrow​$definite the layers and outputs
  $\rightarrow​$definite the loss and accuracy
  $\rightarrow​$definite training operates
  $\rightarrow​$definite the summary

- run the session
  $\rightarrow​$init global variable
  $\rightarrow​$definite the saver to save models
  $\rightarrow​$definite summary filewriter to write the summary
  $\rightarrow​$begin the epoch and get batches
  $\rightarrow​$feed the data and begin to training
  $\rightarrow​$write summary
  $\rightarrow​$save models
  $\rightarrow​$test accuracy