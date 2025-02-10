            if params.method == 'differentialDKTIX':
                torch.save({'epoch': epoch, 'state': model.state_dict(), 'sp': model.scaling_params}, outfile)
            
                        # tmp = torch.load(modelfile)
            # tmp_test = torch.load("./save/checkpoints/CUB/identity_differentialDKTIX_aug_5way_1shot/test.tar")
            # print(any(['scaling_params' in str for str in tmp_test['state'].keys()]))
            if params.method in ['differentialDKTIX']:
                model.scaling_params = tmp['sp']
                print({k: torch.norm(v, p=2).item()/np.sqrt(v.numel()) for k, v in model.scaling_params.items()})
    
        
    def other_forward(self, x1, x2, diag=False, **params):
        # Compute J(x1)
        jac1 = vmap(jacrev(self.fnet_single), (None, 0))(params, x1)
        jac1 = [j.flatten(2) for j in jac1]

        # Compute J(x2)
        jac2 = vmap(jacrev(self.fnet_single), (None, 0))(params, x2)
        jac2 = [j.flatten(2) for j in jac2]

        # Compute J(x1) @ J(x2).T
        result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
        print(result.shape)
        result = result.sum(0)
        if diag:
            return result.diag()
        return result
    
    
    
    
    def forward(self, x1, x2, diag=False, **params):
        x1 = x1.reshape(x1.size(0), 3, 84, 84)
        x2 = x2.reshape(x2.size(0), 3, 84, 84)
        if autodiff:
            jac1T = (self.compute_jacobian_autodiff(x1) * self.sp).T
            jac2T = (self.compute_jacobian_autodiff(x2) * self.sp).T if x1 is not x2 else jac1T
        else:
            jac1T = (self.compute_jacobian(x1) * self.sp).T
            jac2T = (self.compute_jacobian(x2) * self.sp).T if x1 is not x2 else jac1T
            
        # print(self.scaling_param.shape)
        # print(jac1T.shape)
        r1 = jac1T
        r2 = jac2T
        
        if self.normalize :
            r1_norm = r1.norm(dim=0, keepdim=True)
            r1 = r1/r1_norm
            r2_norm = r2.norm(dim=0, keepdim=True)
            r2 = r2/r2_norm
        
        result = r1.T@r2
        if diag:
            return result.diag()
        return result
    
    def compute_jacobian(self, inputs):
        """
        Return the jacobian of a batch of inputs, thanks to the vmap functionality
        """
        jac = vmap(jacrev(self.fnet_single_c), (None, 0))(self.params, inputs)
        jac_values = jac.values()

        reshaped_tensors = []
        for j in jac_values:
            if len(j.shape) >= 3:  # For layers with weights
                # Flatten parameters dimensions and then reshape
                flattened = j.flatten(start_dim=1)  # Flattens to [batch, params]
                reshaped = flattened.T  # Transpose to align dimensions as [params, batch]
                reshaped_tensors.append(reshaped)
            elif len(j.shape) == 2:  # For biases or single parameter components
                reshaped_tensors.append(j.T)  # Simply transpose

        # Concatenate all the reshaped tensors into one large matrix
        return torch.cat(reshaped_tensors, dim=0).T
    
    def compute_jacobian_autodiff(self, inputs):
        """
        Return the jacobian of a batch of inputs, using autodifferentiation
        Useful for when dealing with models using batch normalization or other kind of running statistics
        """
        inputs.requires_grad_(True)
        outputs = self.net(inputs)
        N = sum(p.numel() for p in self.f_net.parameters())
        jac = torch.empty(outputs.size(0), N).to("cuda:0")
        for j in range(outputs.size(0)):
            # print(j)
            grad_y1 = torch.autograd.grad(outputs[j, self.c], self.f_net.parameters(), retain_graph=True, create_graph=True) # We need to create and retain every single graph for the gradient to be able to run through during backprop
            # print_memory_usage()
            flattened_tensors = [t.flatten() for t in grad_y1]
            jac[j] = torch.cat(flattened_tensors)
            # print_memory_usage()
            # if device == "cuda":
            #     torch.cuda.empty_cache()
            #     print_memory_usage()
        return jac