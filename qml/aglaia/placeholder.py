#
#   fit forces
#        # Placeholders for the input/output data
#        #with tf.name_scope('Data'):
#        #    tf_input = tf.placeholder(tf.float32, [None, self.n_coord], name="Coordinates")
#        #    tf_output= tf.placeholder(tf.float32, [None, self.n_coord + 1], name="Energy_forces")
#
#        # Making the descriptor from the Cartesian coordinates
#        #with tf.name_scope('Descriptor'):
#        #    X_des = self.available_descriptors[self.descriptor](in_data, n_atoms=self.n_atoms)
#
#        # Number of features in the descriptor
#        #self.n_features = int(self.n_atoms * (self.n_atoms - 1) * 0.5)
#
#        # Randomly initialisation of the weights and biases
#        #with tf.name_scope('weights'):
#        #    weights, biases = self.__generate_weights(n_out=(1+3*self.n_atoms))
#
#        #    # Log weights for tensorboard
#        #    if self.tensorboard:
#        #        tf.summary.histogram("weights_in", weights[0])
#        #        for ii in range(len(self.hidden_layer_sizes) - 1):
#        #            tf.summary.histogram("weights_hidden", weights[ii + 1])
#        #        tf.summary.histogram("weights_out", weights[-1])
#
#
#        # Calculating the output of the neural net
#        #with tf.name_scope('model'):
#        #    out_NN = self.modelNN(X_des, weights, biases)
#
#        # Obtaining the derivative of the neural net energy wrt cartesian coordinates
#        #with tf.name_scope('grad_ene'):
#        #    ene_NN = tf.slice(out_NN,begin=[0,0], size=[-1,1], name='ene_NN')
#        #    grad_ene_NN = tf.gradients(ene_NN, in_data, name='dEne_dr')[0] * (-1)
#
#        # Calculating the cost function
#        #with tf.name_scope('cost_funct'):
#        #    err_ene_force = tf.square(tf.subtract(out_NN, out_data), name='err2_ene_force')
#        #    err_grad = tf.square(tf.subtract(tf.slice(out_data, begin=[0,1], size=[-1,-1]), grad_ene_NN), name='err2_grad')
#
#        #    cost_ene_force = tf.reduce_mean(err_ene_force, name='cost_ene_force')
#        #    cost_grad = tf.reduce_mean(err_grad, name='cost_grad')
#
#        #    reg_term = self.__reg_term(weights)
#
#        #    cost = cost_ene_force + self.alpha_grad*cost_grad + self.alpha_reg * reg_term
#
#        # Training the network
#        #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_init).minimize(cost)
#
#        #if self.tensorboard:
#        #    cost_summary = tf.summary.scalar('cost', cost)
#
#        # Initialisation of the variables
#        #init = tf.global_variables_initializer()
#        #if self.tensorboard:
#        #    merged_summary = tf.summary.merge_all()
#        #    options = tf.RunOptions()
#        #    options.output_partition_graphs = True
#        #    options.trace_level = tf.RunOptions.SOFTWARE_TRACE
#        #    run_metadata = tf.RunMetadata()
#
#        # Running the graph
#        #with tf.Session() as sess:
#        #    if self.tensorboard:
#        #        summary_writer = tf.summary.FileWriter(logdir=self.board_dir,graph=sess.graph)
#        #    sess.run(init)
#
#        #    for iter in range(self.max_iter):
#        #        # This is the total number of batches in which the training set is divided
#        #        n_batches = int(self.n_samples / self.batch_size)
#        #        # This will be used to calculate the average cost per iteration
#        #        avg_cost = 0
#        #        # Learning over the batches of data
#        #        for i in range(n_batches):
#        #            batch_x = X[i * self.batch_size:(i + 1) * self.batch_size, :]
#        #            batch_y = y[i * self.batch_size:(i + 1) * self.batch_size, :]
#        #            opt, c = sess.run([optimizer, cost], feed_dict={in_data: batch_x, out_data: batch_y})
#        #            avg_cost += c / n_batches
#
#        #            if self.tensorboard:
#        #                if iter % self.print_step == 0:
#        #                    # The options flag is needed to obtain profiling information
#        #                    summary = sess.run(merged_summary, feed_dict={in_data: batch_x, out_data: batch_y}, options=options, run_metadata=run_metadata)
#        #                    summary_writer.add_summary(summary, iter)
#        #                    summary_writer.add_run_metadata(run_metadata, 'iteration %d batch %d' % (iter, i))
#
#        #        self.trainCost.append(avg_cost)
#
#        #    # Saving the weights for later re-use
#        #    self.all_weights = []
#        #    self.all_biases = []
#        #    for ii in range(len(weights)):
#        #        self.all_weights.append(sess.run(weights[ii]))
#        #        self.all_biases.append(sess.run(biases[ii]))
#
#
#    def save(self, path):
#        """
#        Stores a .meta, .index, .data_0000-0001 and a check point file, which can be used to restore
#        the trained model.
#
#        :param path: Path of directory where files are stored. The path is assumed to be absolute
#                     unless there are no forward slashes (/) in the path name.
#        :type path: string
#        """
#
#        if self.is_trained == False:
#            raise Exception("The fit function has not been called yet, so the model can't be saved.")
#
#        # Creating a new graph
#        model_graph = tf.Graph()
#
#        with model_graph.as_default():
#            # Making the placeholder for the data
#            xyz_test = tf.placeholder(tf.float32, [None, self.n_coord], name="Cartesian_coord")
#
#            # Making the descriptor from the Cartesian coordinates
#            X_des = self.available_descriptors[self.descriptor](xyz_test, n_atoms=self.n_atoms)
#
#            # Setting up the trained weights
#            weights = []
#            biases = []
#
#            for ii in range(len(self.all_weights)):
#                weights.append(tf.Variable(self.all_weights[ii], name="weights_restore"))
#                biases.append(tf.Variable(self.all_biases[ii], name="biases_restore"))
#
#            # Calculating the ouputs
#            out_NN = self.modelNN(X_des, weights, biases)
#
#            init = tf.global_variables_initializer()
#
#            # Object needed to save the model
#            all_saver = tf.train.Saver(save_relative_paths=True)
#
#            with tf.Session() as sess:
#                sess.run(init)
#
#                # Saving the graph
#                all_saver.save(sess, dir)
#
#    def load_NN(self, dir):
#        """
#        Function that loads a trained estimator.
#
#        :dir: directory where the .meta, .index, .data_0000-0001 and check point files have been saved.
#        """
#
#        # Inserting the weights into the model
#        with tf.Session() as sess:
#            # Loading a saved graph
#            file = dir + ".meta"
#            saver = tf.train.import_meta_graph(file)
#
#            # The model is loaded in the default graph
#            graph = tf.get_default_graph()
#
#            # Loading the graph of out_NN
#            self.out_NN = graph.get_tensor_by_name("output_node:0")
#            self.in_data = graph.get_tensor_by_name("Cartesian_coord:0")
#
#            saver.restore(sess, dir)
#            sess.run(tf.global_variables_initializer())
#
#        self.loadedModel = True
