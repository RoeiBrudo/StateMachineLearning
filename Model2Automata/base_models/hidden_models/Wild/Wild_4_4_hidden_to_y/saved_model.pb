źŃ
ŃŁ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ňś
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0

NoOpNoOp
č
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ł
valueB B
Ę
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
 
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
 

0
1

0
1
­
metrics
regularization_losses
	variables
non_trainable_variables
 layer_regularization_losses

!layers
	trainable_variables
"layer_metrics
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
#metrics
regularization_losses
	variables
$layer_regularization_losses
%non_trainable_variables

&layers
trainable_variables
'layer_metrics
 
 
 
­
(metrics
regularization_losses
	variables
)layer_regularization_losses
*non_trainable_variables

+layers
trainable_variables
,layer_metrics
 
 
 
­
-metrics
regularization_losses
	variables
.layer_regularization_losses
/non_trainable_variables

0layers
trainable_variables
1layer_metrics
 
 
 
­
2metrics
regularization_losses
	variables
3layer_regularization_losses
4non_trainable_variables

5layers
trainable_variables
6layer_metrics
 
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_input_2Placeholder*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
ë
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2dense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_5097
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ä
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__traced_save_5235
ˇ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_restore_5251Ě
Ť

"__inference_signature_wrapper_5097
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCallŮ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_49492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
Ę
Š
A__inference_dense_1_layer_call_and_return_conditional_losses_5161

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ę
Š
A__inference_dense_1_layer_call_and_return_conditional_losses_4964

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙:::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

m
Q__inference_tf_op_layer_diag_part_1_layer_call_and_return_conditional_losses_5016

inputs
identity`
diag_part_1/kConst*
_output_shapes
: *
dtype0*
value	B : 2
diag_part_1/k{
diag_part_1/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
diag_part_1/padding_valueŻ
diag_part_1MatrixDiagPartV3inputsdiag_part_1/k:output:0"diag_part_1/padding_value:output:0*
T0*
_cloned(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
diag_part_1f
IdentityIdentitydiag_part_1:diagonal:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
š
R
6__inference_tf_op_layer_diag_part_1_layer_call_fn_5205

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_diag_part_1_layer_call_and_return_conditional_losses_50162
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ë
z
N__inference_tf_op_layer_MatMul_1_layer_call_and_return_conditional_losses_5187
inputs_0
inputs_1
identity|
MatMul_1MatMulinputs_0inputs_1*
T0*
_cloned(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

MatMul_1o
IdentityIdentityMatMul_1:product:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
ď
¸
 __inference__traced_restore_5251
file_prefix#
assignvariableop_dense_1_kernel#
assignvariableop_1_dense_1_bias

identity_3˘AssignVariableOp˘AssignVariableOp_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ą
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
RestoreV2/shape_and_slicesş
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_2

Identity_3IdentityIdentity_2:output:0^AssignVariableOp^AssignVariableOp_1*
T0*
_output_shapes
: 2

Identity_3"!

identity_3Identity_3:output:0*
_input_shapes

: ::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ă
x
N__inference_tf_op_layer_MatMul_1_layer_call_and_return_conditional_losses_5000

inputs
inputs_1
identityz
MatMul_1MatMulinputsinputs_1*
T0*
_cloned(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

MatMul_1o
IdentityIdentityMatMul_1:product:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ű

+__inference_functional_5_layer_call_fn_5062
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_50552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
ś
ż
__inference__wrapped_model_4949
input_1
input_27
3functional_5_dense_1_matmul_readvariableop_resource8
4functional_5_dense_1_biasadd_readvariableop_resource
identityĚ
*functional_5/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_5_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*functional_5/dense_1/MatMul/ReadVariableOpł
functional_5/dense_1/MatMulMatMulinput_12functional_5/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
functional_5/dense_1/MatMulË
+functional_5/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_5_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+functional_5/dense_1/BiasAdd/ReadVariableOpŐ
functional_5/dense_1/BiasAddBiasAdd%functional_5/dense_1/MatMul:product:03functional_5/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
functional_5/dense_1/BiasAddż
5functional_5/tf_op_layer_Transpose_1/Transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       27
5functional_5/tf_op_layer_Transpose_1/Transpose_1/perm
0functional_5/tf_op_layer_Transpose_1/Transpose_1	Transpose%functional_5/dense_1/BiasAdd:output:0>functional_5/tf_op_layer_Transpose_1/Transpose_1/perm:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙22
0functional_5/tf_op_layer_Transpose_1/Transpose_1ë
*functional_5/tf_op_layer_MatMul_1/MatMul_1MatMulinput_24functional_5/tf_op_layer_Transpose_1/Transpose_1:y:0*
T0*
_cloned(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2,
*functional_5/tf_op_layer_MatMul_1/MatMul_1Ş
2functional_5/tf_op_layer_diag_part_1/diag_part_1/kConst*
_output_shapes
: *
dtype0*
value	B : 24
2functional_5/tf_op_layer_diag_part_1/diag_part_1/kĹ
>functional_5/tf_op_layer_diag_part_1/diag_part_1/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2@
>functional_5/tf_op_layer_diag_part_1/diag_part_1/padding_valueń
0functional_5/tf_op_layer_diag_part_1/diag_part_1MatrixDiagPartV34functional_5/tf_op_layer_MatMul_1/MatMul_1:product:0;functional_5/tf_op_layer_diag_part_1/diag_part_1/k:output:0Gfunctional_5/tf_op_layer_diag_part_1/diag_part_1/padding_value:output:0*
T0*
_cloned(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙22
0functional_5/tf_op_layer_diag_part_1/diag_part_1
IdentityIdentity;functional_5/tf_op_layer_diag_part_1/diag_part_1:diagonal:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:::P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
Ă
š
F__inference_functional_5_layer_call_and_return_conditional_losses_5025
input_1
input_2
dense_1_4975
dense_1_4977
identity˘dense_1/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1_4975dense_1_4977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_49642!
dense_1/StatefulPartitionedCall¤
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_49862)
'tf_op_layer_Transpose_1/PartitionedCallś
$tf_op_layer_MatMul_1/PartitionedCallPartitionedCallinput_20tf_op_layer_Transpose_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_tf_op_layer_MatMul_1_layer_call_and_return_conditional_losses_50002&
$tf_op_layer_MatMul_1/PartitionedCallĽ
'tf_op_layer_diag_part_1/PartitionedCallPartitionedCall-tf_op_layer_MatMul_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_diag_part_1_layer_call_and_return_conditional_losses_50162)
'tf_op_layer_diag_part_1/PartitionedCall˘
IdentityIdentity0tf_op_layer_diag_part_1/PartitionedCall:output:0 ^dense_1/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
¨
Î
F__inference_functional_5_layer_call_and_return_conditional_losses_5114
inputs_0
inputs_1*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityĽ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulinputs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpĄ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAddĽ
(tf_op_layer_Transpose_1/Transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(tf_op_layer_Transpose_1/Transpose_1/permĺ
#tf_op_layer_Transpose_1/Transpose_1	Transposedense_1/BiasAdd:output:01tf_op_layer_Transpose_1/Transpose_1/perm:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tf_op_layer_Transpose_1/Transpose_1Ĺ
tf_op_layer_MatMul_1/MatMul_1MatMulinputs_1'tf_op_layer_Transpose_1/Transpose_1:y:0*
T0*
_cloned(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
tf_op_layer_MatMul_1/MatMul_1
%tf_op_layer_diag_part_1/diag_part_1/kConst*
_output_shapes
: *
dtype0*
value	B : 2'
%tf_op_layer_diag_part_1/diag_part_1/kŤ
1tf_op_layer_diag_part_1/diag_part_1/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1tf_op_layer_diag_part_1/diag_part_1/padding_value°
#tf_op_layer_diag_part_1/diag_part_1MatrixDiagPartV3'tf_op_layer_MatMul_1/MatMul_1:product:0.tf_op_layer_diag_part_1/diag_part_1/k:output:0:tf_op_layer_diag_part_1/diag_part_1/padding_value:output:0*
T0*
_cloned(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tf_op_layer_diag_part_1/diag_part_1~
IdentityIdentity.tf_op_layer_diag_part_1/diag_part_1:diagonal:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:::Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
¨
Î
F__inference_functional_5_layer_call_and_return_conditional_losses_5131
inputs_0
inputs_1*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityĽ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulinputs_0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpĄ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAddĽ
(tf_op_layer_Transpose_1/Transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2*
(tf_op_layer_Transpose_1/Transpose_1/permĺ
#tf_op_layer_Transpose_1/Transpose_1	Transposedense_1/BiasAdd:output:01tf_op_layer_Transpose_1/Transpose_1/perm:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tf_op_layer_Transpose_1/Transpose_1Ĺ
tf_op_layer_MatMul_1/MatMul_1MatMulinputs_1'tf_op_layer_Transpose_1/Transpose_1:y:0*
T0*
_cloned(*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2
tf_op_layer_MatMul_1/MatMul_1
%tf_op_layer_diag_part_1/diag_part_1/kConst*
_output_shapes
: *
dtype0*
value	B : 2'
%tf_op_layer_diag_part_1/diag_part_1/kŤ
1tf_op_layer_diag_part_1/diag_part_1/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1tf_op_layer_diag_part_1/diag_part_1/padding_value°
#tf_op_layer_diag_part_1/diag_part_1MatrixDiagPartV3'tf_op_layer_MatMul_1/MatMul_1:product:0.tf_op_layer_diag_part_1/diag_part_1/k:output:0:tf_op_layer_diag_part_1/diag_part_1/padding_value:output:0*
T0*
_cloned(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#tf_op_layer_diag_part_1/diag_part_1~
IdentityIdentity.tf_op_layer_diag_part_1/diag_part_1:diagonal:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:::Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
Á
š
F__inference_functional_5_layer_call_and_return_conditional_losses_5055

inputs
inputs_1
dense_1_5046
dense_1_5048
identity˘dense_1/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_5046dense_1_5048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_49642!
dense_1/StatefulPartitionedCall¤
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_49862)
'tf_op_layer_Transpose_1/PartitionedCallˇ
$tf_op_layer_MatMul_1/PartitionedCallPartitionedCallinputs_10tf_op_layer_Transpose_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_tf_op_layer_MatMul_1_layer_call_and_return_conditional_losses_50002&
$tf_op_layer_MatMul_1/PartitionedCallĽ
'tf_op_layer_diag_part_1/PartitionedCallPartitionedCall-tf_op_layer_MatMul_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_diag_part_1_layer_call_and_return_conditional_losses_50162)
'tf_op_layer_diag_part_1/PartitionedCall˘
IdentityIdentity0tf_op_layer_diag_part_1/PartitionedCall:output:0 ^dense_1/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ů
{
&__inference_dense_1_layer_call_fn_5170

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_49642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*.
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ű

+__inference_functional_5_layer_call_fn_5085
input_1
input_2
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_50782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
Ă
š
F__inference_functional_5_layer_call_and_return_conditional_losses_5038
input_1
input_2
dense_1_5029
dense_1_5031
identity˘dense_1/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1_5029dense_1_5031*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_49642!
dense_1/StatefulPartitionedCall¤
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_49862)
'tf_op_layer_Transpose_1/PartitionedCallś
$tf_op_layer_MatMul_1/PartitionedCallPartitionedCallinput_20tf_op_layer_Transpose_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_tf_op_layer_MatMul_1_layer_call_and_return_conditional_losses_50002&
$tf_op_layer_MatMul_1/PartitionedCallĽ
'tf_op_layer_diag_part_1/PartitionedCallPartitionedCall-tf_op_layer_MatMul_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_diag_part_1_layer_call_and_return_conditional_losses_50162)
'tf_op_layer_diag_part_1/PartitionedCall˘
IdentityIdentity0tf_op_layer_diag_part_1/PartitionedCall:output:0 ^dense_1/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_1:PL
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2
ő
Ć
__inference__traced_save_5235
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b0e19a58777149c19672def1538d01e2/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ą
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
: ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 

m
Q__inference_tf_op_layer_diag_part_1_layer_call_and_return_conditional_losses_5200

inputs
identity`
diag_part_1/kConst*
_output_shapes
: *
dtype0*
value	B : 2
diag_part_1/k{
diag_part_1/padding_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
diag_part_1/padding_valueŻ
diag_part_1MatrixDiagPartV3inputsdiag_part_1/k:output:0"diag_part_1/padding_value:output:0*
T0*
_cloned(*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2
diag_part_1f
IdentityIdentitydiag_part_1:diagonal:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ż
R
6__inference_tf_op_layer_Transpose_1_layer_call_fn_5181

inputs
identityŇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_49862
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
á

+__inference_functional_5_layer_call_fn_5141
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_50552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
Á
š
F__inference_functional_5_layer_call_and_return_conditional_losses_5078

inputs
inputs_1
dense_1_5069
dense_1_5071
identity˘dense_1/StatefulPartitionedCall
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_5069dense_1_5071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_49642!
dense_1/StatefulPartitionedCall¤
'tf_op_layer_Transpose_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_49862)
'tf_op_layer_Transpose_1/PartitionedCallˇ
$tf_op_layer_MatMul_1/PartitionedCallPartitionedCallinputs_10tf_op_layer_Transpose_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_tf_op_layer_MatMul_1_layer_call_and_return_conditional_losses_50002&
$tf_op_layer_MatMul_1/PartitionedCallĽ
'tf_op_layer_diag_part_1/PartitionedCallPartitionedCall-tf_op_layer_MatMul_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_tf_op_layer_diag_part_1_layer_call_and_return_conditional_losses_50162)
'tf_op_layer_diag_part_1/PartitionedCall˘
IdentityIdentity0tf_op_layer_diag_part_1/PartitionedCall:output:0 ^dense_1/StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ŕ
_
3__inference_tf_op_layer_MatMul_1_layer_call_fn_5193
inputs_0
inputs_1
identityĺ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_tf_op_layer_MatMul_1_layer_call_and_return_conditional_losses_50002
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1
ń
m
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_4986

inputs
identityu
Transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose_1/perm
Transpose_1	TransposeinputsTranspose_1/perm:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Transpose_1c
IdentityIdentityTranspose_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ń
m
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_5176

inputs
identityu
Transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Transpose_1/perm
Transpose_1	TransposeinputsTranspose_1/perm:output:0*
T0*
_cloned(*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Transpose_1c
IdentityIdentityTranspose_1:y:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*&
_input_shapes
:˙˙˙˙˙˙˙˙˙:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
á

+__inference_functional_5_layer_call_fn_5151
inputs_0
inputs_1
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_functional_5_layer_call_and_return_conditional_losses_50782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*A
_input_shapes0
.:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs/1"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ó
serving_defaultß
;
input_10
serving_default_input_1:0˙˙˙˙˙˙˙˙˙
;
input_20
serving_default_input_2:0˙˙˙˙˙˙˙˙˙G
tf_op_layer_diag_part_1,
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:É
Í*
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer-5
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
7__call__
8_default_save_signature
*9&call_and_return_all_conditional_losses"Š(
_tf_keras_network({"class_name": "Functional", "name": "functional_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Transpose_1", "op": "Transpose", "input": ["dense_1/BiasAdd", "Transpose_1/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MatMul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "MatMul_1", "op": "MatMul", "input": ["input_2", "Transpose_1"], "attr": {"transpose_b": {"b": false}, "T": {"type": "DT_FLOAT"}, "transpose_a": {"b": false}}}, "constants": {}}, "name": "tf_op_layer_MatMul_1", "inbound_nodes": [[["input_2", 0, 0, {}], ["tf_op_layer_Transpose_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "diag_part_1", "trainable": true, "dtype": "float32", "node_def": {"name": "diag_part_1", "op": "MatrixDiagPartV3", "input": ["MatMul_1", "diag_part_1/k", "diag_part_1/padding_value"], "attr": {"T": {"type": "DT_FLOAT"}, "align": {"s": "UklHSFRfTEVGVA=="}}}, "constants": {"1": 0, "2": 0.0}}, "name": "tf_op_layer_diag_part_1", "inbound_nodes": [[["tf_op_layer_MatMul_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["tf_op_layer_diag_part_1", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 4]}, {"class_name": "TensorShape", "items": [null, 4]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "Transpose_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Transpose_1", "op": "Transpose", "input": ["dense_1/BiasAdd", "Transpose_1/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 0]}}, "name": "tf_op_layer_Transpose_1", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "MatMul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "MatMul_1", "op": "MatMul", "input": ["input_2", "Transpose_1"], "attr": {"transpose_b": {"b": false}, "T": {"type": "DT_FLOAT"}, "transpose_a": {"b": false}}}, "constants": {}}, "name": "tf_op_layer_MatMul_1", "inbound_nodes": [[["input_2", 0, 0, {}], ["tf_op_layer_Transpose_1", 0, 0, {}]]]}, {"class_name": "TensorFlowOpLayer", "config": {"name": "diag_part_1", "trainable": true, "dtype": "float32", "node_def": {"name": "diag_part_1", "op": "MatrixDiagPartV3", "input": ["MatMul_1", "diag_part_1/k", "diag_part_1/padding_value"], "attr": {"T": {"type": "DT_FLOAT"}, "align": {"s": "UklHSFRfTEVGVA=="}}}, "constants": {"1": 0, "2": 0.0}}, "name": "tf_op_layer_diag_part_1", "inbound_nodes": [[["tf_op_layer_MatMul_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["tf_op_layer_diag_part_1", 0, 0]]}}}
é"ć
_tf_keras_input_layerĆ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ď

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
:__call__
*;&call_and_return_all_conditional_losses"Ę
_tf_keras_layer°{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
é"ć
_tf_keras_input_layerĆ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}

regularization_losses
	variables
trainable_variables
	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layerń{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_Transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Transpose_1", "trainable": true, "dtype": "float32", "node_def": {"name": "Transpose_1", "op": "Transpose", "input": ["dense_1/BiasAdd", "Transpose_1/perm"], "attr": {"Tperm": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": [1, 0]}}}

regularization_losses
	variables
trainable_variables
	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layerč{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_MatMul_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "MatMul_1", "trainable": true, "dtype": "float32", "node_def": {"name": "MatMul_1", "op": "MatMul", "input": ["input_2", "Transpose_1"], "attr": {"transpose_b": {"b": false}, "T": {"type": "DT_FLOAT"}, "transpose_a": {"b": false}}}, "constants": {}}}
ž
regularization_losses
	variables
trainable_variables
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"Ż
_tf_keras_layer{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_diag_part_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "diag_part_1", "trainable": true, "dtype": "float32", "node_def": {"name": "diag_part_1", "op": "MatrixDiagPartV3", "input": ["MatMul_1", "diag_part_1/k", "diag_part_1/padding_value"], "attr": {"T": {"type": "DT_FLOAT"}, "align": {"s": "UklHSFRfTEVGVA=="}}}, "constants": {"1": 0, "2": 0.0}}}
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ę
metrics
regularization_losses
	variables
non_trainable_variables
 layer_regularization_losses

!layers
	trainable_variables
"layer_metrics
7__call__
8_default_save_signature
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
,
Bserving_default"
signature_map
 :2dense_1/kernel
:2dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
#metrics
regularization_losses
	variables
$layer_regularization_losses
%non_trainable_variables

&layers
trainable_variables
'layer_metrics
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
(metrics
regularization_losses
	variables
)layer_regularization_losses
*non_trainable_variables

+layers
trainable_variables
,layer_metrics
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
-metrics
regularization_losses
	variables
.layer_regularization_losses
/non_trainable_variables

0layers
trainable_variables
1layer_metrics
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
2metrics
regularization_losses
	variables
3layer_regularization_losses
4non_trainable_variables

5layers
trainable_variables
6layer_metrics
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ú2÷
+__inference_functional_5_layer_call_fn_5151
+__inference_functional_5_layer_call_fn_5062
+__inference_functional_5_layer_call_fn_5141
+__inference_functional_5_layer_call_fn_5085Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
2
__inference__wrapped_model_4949Ţ
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *N˘K
IF
!
input_1˙˙˙˙˙˙˙˙˙
!
input_2˙˙˙˙˙˙˙˙˙
ć2ă
F__inference_functional_5_layer_call_and_return_conditional_losses_5038
F__inference_functional_5_layer_call_and_return_conditional_losses_5114
F__inference_functional_5_layer_call_and_return_conditional_losses_5025
F__inference_functional_5_layer_call_and_return_conditional_losses_5131Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Đ2Í
&__inference_dense_1_layer_call_fn_5170˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ë2č
A__inference_dense_1_layer_call_and_return_conditional_losses_5161˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ŕ2Ý
6__inference_tf_op_layer_Transpose_1_layer_call_fn_5181˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ű2ř
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_5176˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ý2Ú
3__inference_tf_op_layer_MatMul_1_layer_call_fn_5193˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ř2ő
N__inference_tf_op_layer_MatMul_1_layer_call_and_return_conditional_losses_5187˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ŕ2Ý
6__inference_tf_op_layer_diag_part_1_layer_call_fn_5205˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ű2ř
Q__inference_tf_op_layer_diag_part_1_layer_call_and_return_conditional_losses_5200˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
8B6
"__inference_signature_wrapper_5097input_1input_2Ń
__inference__wrapped_model_4949­X˘U
N˘K
IF
!
input_1˙˙˙˙˙˙˙˙˙
!
input_2˙˙˙˙˙˙˙˙˙
Ş "MŞJ
H
tf_op_layer_diag_part_1-*
tf_op_layer_diag_part_1˙˙˙˙˙˙˙˙˙Ą
A__inference_dense_1_layer_call_and_return_conditional_losses_5161\/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 y
&__inference_dense_1_layer_call_fn_5170O/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ô
F__inference_functional_5_layer_call_and_return_conditional_losses_5025`˘]
V˘S
IF
!
input_1˙˙˙˙˙˙˙˙˙
!
input_2˙˙˙˙˙˙˙˙˙
p

 
Ş "!˘

0˙˙˙˙˙˙˙˙˙
 Ô
F__inference_functional_5_layer_call_and_return_conditional_losses_5038`˘]
V˘S
IF
!
input_1˙˙˙˙˙˙˙˙˙
!
input_2˙˙˙˙˙˙˙˙˙
p 

 
Ş "!˘

0˙˙˙˙˙˙˙˙˙
 Ö
F__inference_functional_5_layer_call_and_return_conditional_losses_5114b˘_
X˘U
KH
"
inputs/0˙˙˙˙˙˙˙˙˙
"
inputs/1˙˙˙˙˙˙˙˙˙
p

 
Ş "!˘

0˙˙˙˙˙˙˙˙˙
 Ö
F__inference_functional_5_layer_call_and_return_conditional_losses_5131b˘_
X˘U
KH
"
inputs/0˙˙˙˙˙˙˙˙˙
"
inputs/1˙˙˙˙˙˙˙˙˙
p 

 
Ş "!˘

0˙˙˙˙˙˙˙˙˙
 Ť
+__inference_functional_5_layer_call_fn_5062|`˘]
V˘S
IF
!
input_1˙˙˙˙˙˙˙˙˙
!
input_2˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙Ť
+__inference_functional_5_layer_call_fn_5085|`˘]
V˘S
IF
!
input_1˙˙˙˙˙˙˙˙˙
!
input_2˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙­
+__inference_functional_5_layer_call_fn_5141~b˘_
X˘U
KH
"
inputs/0˙˙˙˙˙˙˙˙˙
"
inputs/1˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙­
+__inference_functional_5_layer_call_fn_5151~b˘_
X˘U
KH
"
inputs/0˙˙˙˙˙˙˙˙˙
"
inputs/1˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙ĺ
"__inference_signature_wrapper_5097ži˘f
˘ 
_Ş\
,
input_1!
input_1˙˙˙˙˙˙˙˙˙
,
input_2!
input_2˙˙˙˙˙˙˙˙˙"MŞJ
H
tf_op_layer_diag_part_1-*
tf_op_layer_diag_part_1˙˙˙˙˙˙˙˙˙ß
N__inference_tf_op_layer_MatMul_1_layer_call_and_return_conditional_losses_5187Z˘W
P˘M
KH
"
inputs/0˙˙˙˙˙˙˙˙˙
"
inputs/1˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 ś
3__inference_tf_op_layer_MatMul_1_layer_call_fn_5193Z˘W
P˘M
KH
"
inputs/0˙˙˙˙˙˙˙˙˙
"
inputs/1˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙­
Q__inference_tf_op_layer_Transpose_1_layer_call_and_return_conditional_losses_5176X/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
6__inference_tf_op_layer_Transpose_1_layer_call_fn_5181K/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙˛
Q__inference_tf_op_layer_diag_part_1_layer_call_and_return_conditional_losses_5200]8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "!˘

0˙˙˙˙˙˙˙˙˙
 
6__inference_tf_op_layer_diag_part_1_layer_call_fn_5205P8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙