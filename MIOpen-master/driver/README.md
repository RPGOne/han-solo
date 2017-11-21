# MIOpen Driver

The driver enables to test forward/backward of any particular layer in MIOpen. 

To build the driver, type `make MIOpenDriver` from the `build` directory.

All the supported layers in MIOpen can be found by the supported `base_args` here:

` ./bin/MIOpenDriver --help `

To execute from the build directory: `./bin/MIOpenDriver *base_arg* *layer_specific_args*`

Sample runs:
* Convoluton with search on - 
`./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2` 
* Forward convolution with search off - 
`./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1`
* Pooling with default parameters
`./bin/MIOpenDriver pool`
* LRN with default parameters and timing on -
`./bin/MIOpenDriver lrn -t 1`
* Batch normalization with spatial norm forward training, saving mean and variance tensors
`./bin/MIOpenDriver bnorm -F 1 -n 32 -c 512 -H 16 -W 16 -m 1 -s 1`
* Printout layer specific input arguments -
`./bin/MIOpenDriver *base_arg* -?` or `./bin/MIOpenDriver *base_arg* -h (--help)`

