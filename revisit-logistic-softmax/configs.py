save_dir                    = './save/'
data_dir = {}
data_dir['CUB']             = './filelists/CUB/'
data_dir['miniImagenet']    = './filelists/miniImagenet/'
data_dir['omniglot']        = './filelists/omniglot/'
kernel_type                 = 'NTK' #linear, rbf, spectral (regression only), matern, poli1, poli2, cossim, bncossim, NTK, cossimNTK

autodiff                    = False # Using the autodiff of pytorch to compute the jacobian