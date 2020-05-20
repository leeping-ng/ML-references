# Summary of Non-Eager Execution


### Overview
Writing and running programs in TensorFlow has the following steps:

1. Create Tensors (variables) that are not yet executed/evaluated.
2. Write operations between those Tensors.
3. Initialize your Tensors.
4. Create a Session.
5. Run the Session. This will run the operations you'd written above.

### Placeholders/feed_dict
- When you specify the operations needed for a computation, you are telling TensorFlow how to construct a computation graph. 
- The computation graph can have some placeholders whose values you will specify only later. 
- Finally, when you run the session, you are telling TensorFlow to execute the computation graph.
- Examples of placeholders are the batch to be used, or the dropout keep_prob
