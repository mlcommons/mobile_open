
I
inputPlaceholder*&
shape:???????????*
dtype0
2
MobilenetEdgeTPU/inputIdentityinput*
T0
?
@MobilenetEdgeTPU/Conv/weights/Initializer/truncated_normal/shapeConst*%
valueB"             *0
_class&
$"loc:@MobilenetEdgeTPU/Conv/weights*
dtype0
?
?MobilenetEdgeTPU/Conv/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *0
_class&
$"loc:@MobilenetEdgeTPU/Conv/weights*
dtype0
?
AMobilenetEdgeTPU/Conv/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*0
_class&
$"loc:@MobilenetEdgeTPU/Conv/weights*
dtype0
?
JMobilenetEdgeTPU/Conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal@MobilenetEdgeTPU/Conv/weights/Initializer/truncated_normal/shape*0
_class&
$"loc:@MobilenetEdgeTPU/Conv/weights*
dtype0*
seed2 *
T0*

seed 
?
>MobilenetEdgeTPU/Conv/weights/Initializer/truncated_normal/mulMulJMobilenetEdgeTPU/Conv/weights/Initializer/truncated_normal/TruncatedNormalAMobilenetEdgeTPU/Conv/weights/Initializer/truncated_normal/stddev*0
_class&
$"loc:@MobilenetEdgeTPU/Conv/weights*
T0
?
:MobilenetEdgeTPU/Conv/weights/Initializer/truncated_normalAdd>MobilenetEdgeTPU/Conv/weights/Initializer/truncated_normal/mul?MobilenetEdgeTPU/Conv/weights/Initializer/truncated_normal/mean*
T0*0
_class&
$"loc:@MobilenetEdgeTPU/Conv/weights
?
MobilenetEdgeTPU/Conv/weights
VariableV2*
shape: *
	container *0
_class&
$"loc:@MobilenetEdgeTPU/Conv/weights*
dtype0*
shared_name 
?
$MobilenetEdgeTPU/Conv/weights/AssignAssignMobilenetEdgeTPU/Conv/weights:MobilenetEdgeTPU/Conv/weights/Initializer/truncated_normal*
T0*
validate_shape(*0
_class&
$"loc:@MobilenetEdgeTPU/Conv/weights*
use_locking(
?
"MobilenetEdgeTPU/Conv/weights/readIdentityMobilenetEdgeTPU/Conv/weights*
T0*0
_class&
$"loc:@MobilenetEdgeTPU/Conv/weights
X
#MobilenetEdgeTPU/Conv/dilation_rateConst*
valueB"      *
dtype0
?
MobilenetEdgeTPU/Conv/Conv2DConv2DMobilenetEdgeTPU/input"MobilenetEdgeTPU/Conv/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
6MobilenetEdgeTPU/Conv/BatchNorm/gamma/Initializer/onesConst*
valueB *  ??*8
_class.
,*loc:@MobilenetEdgeTPU/Conv/BatchNorm/gamma*
dtype0
?
%MobilenetEdgeTPU/Conv/BatchNorm/gamma
VariableV2*8
_class.
,*loc:@MobilenetEdgeTPU/Conv/BatchNorm/gamma*
dtype0*
shared_name *
shape: *
	container 
?
,MobilenetEdgeTPU/Conv/BatchNorm/gamma/AssignAssign%MobilenetEdgeTPU/Conv/BatchNorm/gamma6MobilenetEdgeTPU/Conv/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*8
_class.
,*loc:@MobilenetEdgeTPU/Conv/BatchNorm/gamma*
use_locking(
?
*MobilenetEdgeTPU/Conv/BatchNorm/gamma/readIdentity%MobilenetEdgeTPU/Conv/BatchNorm/gamma*8
_class.
,*loc:@MobilenetEdgeTPU/Conv/BatchNorm/gamma*
T0
?
6MobilenetEdgeTPU/Conv/BatchNorm/beta/Initializer/zerosConst*
valueB *    *7
_class-
+)loc:@MobilenetEdgeTPU/Conv/BatchNorm/beta*
dtype0
?
$MobilenetEdgeTPU/Conv/BatchNorm/beta
VariableV2*
shape: *
	container *7
_class-
+)loc:@MobilenetEdgeTPU/Conv/BatchNorm/beta*
dtype0*
shared_name 
?
+MobilenetEdgeTPU/Conv/BatchNorm/beta/AssignAssign$MobilenetEdgeTPU/Conv/BatchNorm/beta6MobilenetEdgeTPU/Conv/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*7
_class-
+)loc:@MobilenetEdgeTPU/Conv/BatchNorm/beta*
use_locking(
?
)MobilenetEdgeTPU/Conv/BatchNorm/beta/readIdentity$MobilenetEdgeTPU/Conv/BatchNorm/beta*
T0*7
_class-
+)loc:@MobilenetEdgeTPU/Conv/BatchNorm/beta
?
=MobilenetEdgeTPU/Conv/BatchNorm/moving_mean/Initializer/zerosConst*
valueB *    *>
_class4
20loc:@MobilenetEdgeTPU/Conv/BatchNorm/moving_mean*
dtype0
?
+MobilenetEdgeTPU/Conv/BatchNorm/moving_mean
VariableV2*
shape: *
	container *>
_class4
20loc:@MobilenetEdgeTPU/Conv/BatchNorm/moving_mean*
dtype0*
shared_name 
?
2MobilenetEdgeTPU/Conv/BatchNorm/moving_mean/AssignAssign+MobilenetEdgeTPU/Conv/BatchNorm/moving_mean=MobilenetEdgeTPU/Conv/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*>
_class4
20loc:@MobilenetEdgeTPU/Conv/BatchNorm/moving_mean*
use_locking(
?
0MobilenetEdgeTPU/Conv/BatchNorm/moving_mean/readIdentity+MobilenetEdgeTPU/Conv/BatchNorm/moving_mean*
T0*>
_class4
20loc:@MobilenetEdgeTPU/Conv/BatchNorm/moving_mean
?
@MobilenetEdgeTPU/Conv/BatchNorm/moving_variance/Initializer/onesConst*
valueB *  ??*B
_class8
64loc:@MobilenetEdgeTPU/Conv/BatchNorm/moving_variance*
dtype0
?
/MobilenetEdgeTPU/Conv/BatchNorm/moving_variance
VariableV2*
shape: *
	container *B
_class8
64loc:@MobilenetEdgeTPU/Conv/BatchNorm/moving_variance*
dtype0*
shared_name 
?
6MobilenetEdgeTPU/Conv/BatchNorm/moving_variance/AssignAssign/MobilenetEdgeTPU/Conv/BatchNorm/moving_variance@MobilenetEdgeTPU/Conv/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*B
_class8
64loc:@MobilenetEdgeTPU/Conv/BatchNorm/moving_variance*
use_locking(
?
4MobilenetEdgeTPU/Conv/BatchNorm/moving_variance/readIdentity/MobilenetEdgeTPU/Conv/BatchNorm/moving_variance*
T0*B
_class8
64loc:@MobilenetEdgeTPU/Conv/BatchNorm/moving_variance
?
0MobilenetEdgeTPU/Conv/BatchNorm/FusedBatchNormV3FusedBatchNormV3MobilenetEdgeTPU/Conv/Conv2D*MobilenetEdgeTPU/Conv/BatchNorm/gamma/read)MobilenetEdgeTPU/Conv/BatchNorm/beta/read0MobilenetEdgeTPU/Conv/BatchNorm/moving_mean/read4MobilenetEdgeTPU/Conv/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
R
%MobilenetEdgeTPU/Conv/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
]
MobilenetEdgeTPU/Conv/ReluRelu0MobilenetEdgeTPU/Conv/BatchNorm/FusedBatchNormV3*
T0
U
$MobilenetEdgeTPU/expanded_conv/inputIdentityMobilenetEdgeTPU/Conv/Relu*
T0
?
QMobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"             *A
_class7
53loc:@MobilenetEdgeTPU/expanded_conv/project/weights*
dtype0
?
PMobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@MobilenetEdgeTPU/expanded_conv/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*A
_class7
53loc:@MobilenetEdgeTPU/expanded_conv/project/weights*
dtype0
?
[MobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQMobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normal/shape*A
_class7
53loc:@MobilenetEdgeTPU/expanded_conv/project/weights*
dtype0*
seed2 *
T0*

seed 
?
OMobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normal/mulMul[MobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normal/TruncatedNormalRMobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@MobilenetEdgeTPU/expanded_conv/project/weights
?
KMobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normalAddOMobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normal/mulPMobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normal/mean*A
_class7
53loc:@MobilenetEdgeTPU/expanded_conv/project/weights*
T0
?
.MobilenetEdgeTPU/expanded_conv/project/weights
VariableV2*
shared_name *
shape: *
	container *A
_class7
53loc:@MobilenetEdgeTPU/expanded_conv/project/weights*
dtype0
?
5MobilenetEdgeTPU/expanded_conv/project/weights/AssignAssign.MobilenetEdgeTPU/expanded_conv/project/weightsKMobilenetEdgeTPU/expanded_conv/project/weights/Initializer/truncated_normal*
T0*
validate_shape(*A
_class7
53loc:@MobilenetEdgeTPU/expanded_conv/project/weights*
use_locking(
?
3MobilenetEdgeTPU/expanded_conv/project/weights/readIdentity.MobilenetEdgeTPU/expanded_conv/project/weights*
T0*A
_class7
53loc:@MobilenetEdgeTPU/expanded_conv/project/weights
i
4MobilenetEdgeTPU/expanded_conv/project/dilation_rateConst*
dtype0*
valueB"      
?
-MobilenetEdgeTPU/expanded_conv/project/Conv2DConv2D$MobilenetEdgeTPU/expanded_conv/input3MobilenetEdgeTPU/expanded_conv/project/weights/read*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 
?
GMobilenetEdgeTPU/expanded_conv/project/BatchNorm/gamma/Initializer/onesConst*
dtype0*
valueB*  ??*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/gamma
?
6MobilenetEdgeTPU/expanded_conv/project/BatchNorm/gamma
VariableV2*
shape:*
	container *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/gamma*
dtype0*
shared_name 
?
=MobilenetEdgeTPU/expanded_conv/project/BatchNorm/gamma/AssignAssign6MobilenetEdgeTPU/expanded_conv/project/BatchNorm/gammaGMobilenetEdgeTPU/expanded_conv/project/BatchNorm/gamma/Initializer/ones*
validate_shape(*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/gamma*
use_locking(*
T0
?
;MobilenetEdgeTPU/expanded_conv/project/BatchNorm/gamma/readIdentity6MobilenetEdgeTPU/expanded_conv/project/BatchNorm/gamma*
T0*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/gamma
?
GMobilenetEdgeTPU/expanded_conv/project/BatchNorm/beta/Initializer/zerosConst*
valueB*    *H
_class>
<:loc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/beta*
dtype0
?
5MobilenetEdgeTPU/expanded_conv/project/BatchNorm/beta
VariableV2*
shape:*
	container *H
_class>
<:loc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/beta*
dtype0*
shared_name 
?
<MobilenetEdgeTPU/expanded_conv/project/BatchNorm/beta/AssignAssign5MobilenetEdgeTPU/expanded_conv/project/BatchNorm/betaGMobilenetEdgeTPU/expanded_conv/project/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*H
_class>
<:loc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/beta*
use_locking(
?
:MobilenetEdgeTPU/expanded_conv/project/BatchNorm/beta/readIdentity5MobilenetEdgeTPU/expanded_conv/project/BatchNorm/beta*H
_class>
<:loc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/beta*
T0
?
NMobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_mean/Initializer/zerosConst*O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_mean*
dtype0*
valueB*    
?
<MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_mean
VariableV2*
shape:*
	container *O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
CMobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_mean/AssignAssign<MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_meanNMobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_mean*
use_locking(
?
AMobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_mean/readIdentity<MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_mean*
T0*O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_mean
?
QMobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
valueB*  ??*S
_classI
GEloc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_variance
?
@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_variance
VariableV2*S
_classI
GEloc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_variance*
dtype0*
shared_name *
shape:*
	container 
?
GMobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_variance/AssignAssign@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_varianceQMobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*S
_classI
GEloc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_variance*
use_locking(
?
EMobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_variance/readIdentity@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_variance*S
_classI
GEloc:@MobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_variance*
T0
?
AMobilenetEdgeTPU/expanded_conv/project/BatchNorm/FusedBatchNormV3FusedBatchNormV3-MobilenetEdgeTPU/expanded_conv/project/Conv2D;MobilenetEdgeTPU/expanded_conv/project/BatchNorm/gamma/read:MobilenetEdgeTPU/expanded_conv/project/BatchNorm/beta/readAMobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_mean/readEMobilenetEdgeTPU/expanded_conv/project/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
c
6MobilenetEdgeTPU/expanded_conv/project/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
/MobilenetEdgeTPU/expanded_conv/project/IdentityIdentityAMobilenetEdgeTPU/expanded_conv/project/BatchNorm/FusedBatchNormV3*
T0
k
%MobilenetEdgeTPU/expanded_conv/outputIdentity/MobilenetEdgeTPU/expanded_conv/project/Identity*
T0
b
&MobilenetEdgeTPU/expanded_conv_1/inputIdentity%MobilenetEdgeTPU/expanded_conv/output*
T0
?
RMobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"         ?   *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_1/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_1/expand/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_1/expand/weights*
dtype0
?
\MobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRMobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_1/expand/weights*
dtype0
?
PMobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normal/mulMul\MobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normal/TruncatedNormalSMobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_1/expand/weights
?
LMobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normalAddPMobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normal/mulQMobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_1/expand/weights
?
/MobilenetEdgeTPU/expanded_conv_1/expand/weights
VariableV2*
shape:?*
	container *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_1/expand/weights*
dtype0*
shared_name 
?
6MobilenetEdgeTPU/expanded_conv_1/expand/weights/AssignAssign/MobilenetEdgeTPU/expanded_conv_1/expand/weightsLMobilenetEdgeTPU/expanded_conv_1/expand/weights/Initializer/truncated_normal*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_1/expand/weights*
use_locking(*
T0*
validate_shape(
?
4MobilenetEdgeTPU/expanded_conv_1/expand/weights/readIdentity/MobilenetEdgeTPU/expanded_conv_1/expand/weights*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_1/expand/weights
j
5MobilenetEdgeTPU/expanded_conv_1/expand/dilation_rateConst*
dtype0*
valueB"      
?
.MobilenetEdgeTPU/expanded_conv_1/expand/Conv2DConv2D&MobilenetEdgeTPU/expanded_conv_1/input4MobilenetEdgeTPU/expanded_conv_1/expand/weights/read*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
strides
*
T0*
data_formatNHWC
?
HMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gamma*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gamma
VariableV2*
dtype0*
shared_name *
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gamma
?
>MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gamma/AssignAssign7MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gammaHMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gamma*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gamma/readIdentity7MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gamma*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gamma*
T0
?
HMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/beta*
dtype0
?
6MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/beta
VariableV2*
shape:?*
	container *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/beta*
dtype0*
shared_name 
?
=MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/beta/AssignAssign6MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/betaHMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*
validate_shape(*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/beta
?
;MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/beta/readIdentity6MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/beta*
T0*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/beta
?
OMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_mean
?
=MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_mean*
dtype0*
shared_name 
?
DMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_mean/AssignAssign=MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_meanOMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_mean*
use_locking(
?
BMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_mean/readIdentity=MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_mean*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_mean*
T0
?
RMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_variance*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_variance/AssignAssignAMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_varianceRMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_variance/readIdentityAMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_variance*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_variance
?
BMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3.MobilenetEdgeTPU/expanded_conv_1/expand/Conv2D<MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/gamma/read;MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/beta/readBMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_mean/readFMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
d
7MobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
,MobilenetEdgeTPU/expanded_conv_1/expand/ReluReluBMobilenetEdgeTPU/expanded_conv_1/expand/BatchNorm/FusedBatchNormV3*
T0
t
1MobilenetEdgeTPU/expanded_conv_1/expansion_outputIdentity,MobilenetEdgeTPU/expanded_conv_1/expand/Relu*
T0
?
SMobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"      ?       *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_1/project/weights
?
RMobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_1/project/weights
?
TMobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_1/project/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normal/shape*
dtype0*
seed2 *
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_1/project/weights
?
QMobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_1/project/weights
?
MMobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normal/mean*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_1/project/weights*
T0
?
0MobilenetEdgeTPU/expanded_conv_1/project/weights
VariableV2*
shape:? *
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_1/project/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_1/project/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_1/project/weightsMMobilenetEdgeTPU/expanded_conv_1/project/weights/Initializer/truncated_normal*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_1/project/weights*
use_locking(*
T0
?
5MobilenetEdgeTPU/expanded_conv_1/project/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_1/project/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_1/project/weights
k
6MobilenetEdgeTPU/expanded_conv_1/project/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_1/project/Conv2DConv2D1MobilenetEdgeTPU/expanded_conv_1/expansion_output5MobilenetEdgeTPU/expanded_conv_1/project/weights/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
IMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gamma/Initializer/onesConst*
valueB *  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gamma
VariableV2*
shape: *
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gamma*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gamma*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gamma
?
IMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/beta/Initializer/zerosConst*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/beta*
dtype0*
valueB *    
?
7MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/beta
VariableV2*
shape: *
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/beta*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB *    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_mean
VariableV2*
shared_name *
shape: *
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_mean*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_mean*
use_locking(*
T0
?
CMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB *  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_variance
VariableV2*
shape: *
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_variance*
use_locking(
?
GMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_variance*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_variance*
T0
?
CMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_1/project/Conv2D=MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
e
8MobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
1MobilenetEdgeTPU/expanded_conv_1/project/IdentityIdentityCMobilenetEdgeTPU/expanded_conv_1/project/BatchNorm/FusedBatchNormV3*
T0
o
'MobilenetEdgeTPU/expanded_conv_1/outputIdentity1MobilenetEdgeTPU/expanded_conv_1/project/Identity*
T0
d
&MobilenetEdgeTPU/expanded_conv_2/inputIdentity'MobilenetEdgeTPU/expanded_conv_1/output*
T0
?
RMobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"          ?   *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_2/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normal/meanConst*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_2/expand/weights*
dtype0*
valueB
 *    
?
SMobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_2/expand/weights*
dtype0
?
\MobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRMobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normal/shape*
T0*

seed *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_2/expand/weights*
dtype0*
seed2 
?
PMobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normal/mulMul\MobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normal/TruncatedNormalSMobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normal/stddev*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_2/expand/weights*
T0
?
LMobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normalAddPMobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normal/mulQMobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_2/expand/weights
?
/MobilenetEdgeTPU/expanded_conv_2/expand/weights
VariableV2*
shape: ?*
	container *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_2/expand/weights*
dtype0*
shared_name 
?
6MobilenetEdgeTPU/expanded_conv_2/expand/weights/AssignAssign/MobilenetEdgeTPU/expanded_conv_2/expand/weightsLMobilenetEdgeTPU/expanded_conv_2/expand/weights/Initializer/truncated_normal*
T0*
validate_shape(*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_2/expand/weights*
use_locking(
?
4MobilenetEdgeTPU/expanded_conv_2/expand/weights/readIdentity/MobilenetEdgeTPU/expanded_conv_2/expand/weights*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_2/expand/weights*
T0
j
5MobilenetEdgeTPU/expanded_conv_2/expand/dilation_rateConst*
dtype0*
valueB"      
?
.MobilenetEdgeTPU/expanded_conv_2/expand/Conv2DConv2D&MobilenetEdgeTPU/expanded_conv_2/input4MobilenetEdgeTPU/expanded_conv_2/expand/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
HMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gamma*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gamma/AssignAssign7MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gammaHMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gamma/Initializer/ones*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gamma*
use_locking(*
T0
?
<MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gamma/readIdentity7MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gamma*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gamma
?
HMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/beta*
dtype0
?
6MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/beta
VariableV2*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/beta*
dtype0*
shared_name *
shape:?*
	container 
?
=MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/beta/AssignAssign6MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/betaHMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/beta/Initializer/zeros*
validate_shape(*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/beta*
use_locking(*
T0
?
;MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/beta/readIdentity6MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/beta*
T0*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/beta
?
OMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_mean*
dtype0
?
=MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_mean
VariableV2*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_mean*
dtype0*
shared_name *
shape:?*
	container 
?
DMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_mean/AssignAssign=MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_meanOMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_mean*
use_locking(*
T0
?
BMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_mean/readIdentity=MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_mean*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_mean
?
RMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_variance*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_variance/AssignAssignAMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_varianceRMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_variance/Initializer/ones*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_variance*
use_locking(*
T0*
validate_shape(
?
FMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_variance/readIdentityAMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_variance*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_variance
?
BMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3.MobilenetEdgeTPU/expanded_conv_2/expand/Conv2D<MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/gamma/read;MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/beta/readBMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_mean/readFMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
d
7MobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
,MobilenetEdgeTPU/expanded_conv_2/expand/ReluReluBMobilenetEdgeTPU/expanded_conv_2/expand/BatchNorm/FusedBatchNormV3*
T0
t
1MobilenetEdgeTPU/expanded_conv_2/expansion_outputIdentity,MobilenetEdgeTPU/expanded_conv_2/expand/Relu*
T0
?
SMobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?       *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_2/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_2/project/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_2/project/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normal/shape*
dtype0*
seed2 *
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_2/project/weights
?
QMobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_2/project/weights
?
MMobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normal/mean*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_2/project/weights*
T0
?
0MobilenetEdgeTPU/expanded_conv_2/project/weights
VariableV2*
dtype0*
shared_name *
shape:? *
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_2/project/weights
?
7MobilenetEdgeTPU/expanded_conv_2/project/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_2/project/weightsMMobilenetEdgeTPU/expanded_conv_2/project/weights/Initializer/truncated_normal*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_2/project/weights*
use_locking(*
T0
?
5MobilenetEdgeTPU/expanded_conv_2/project/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_2/project/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_2/project/weights
k
6MobilenetEdgeTPU/expanded_conv_2/project/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_2/project/Conv2DConv2D1MobilenetEdgeTPU/expanded_conv_2/expansion_output5MobilenetEdgeTPU/expanded_conv_2/project/weights/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
IMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gamma/Initializer/onesConst*
valueB *  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gamma
VariableV2*
shape: *
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gamma*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gamma*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gamma*
T0
?
IMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/beta/Initializer/zerosConst*
valueB *    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/beta
VariableV2*
dtype0*
shared_name *
shape: *
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/beta
?
>MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/beta*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/beta*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/beta*
T0
?
PMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB *    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_mean
VariableV2*
shape: *
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
EMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_mean*
use_locking(
?
CMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB *  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_variance
VariableV2*
shared_name *
shape: *
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_variance*
dtype0
?
IMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_variance*
use_locking(*
T0
?
GMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_variance*
T0*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_variance
?
CMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_2/project/Conv2D=MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/moving_variance/read*
epsilon%o?:*
U0*
T0*
data_formatNHWC*
is_training( 
e
8MobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
1MobilenetEdgeTPU/expanded_conv_2/project/IdentityIdentityCMobilenetEdgeTPU/expanded_conv_2/project/BatchNorm/FusedBatchNormV3*
T0
?
$MobilenetEdgeTPU/expanded_conv_2/addAddV21MobilenetEdgeTPU/expanded_conv_2/project/Identity&MobilenetEdgeTPU/expanded_conv_2/input*
T0
b
'MobilenetEdgeTPU/expanded_conv_2/outputIdentity$MobilenetEdgeTPU/expanded_conv_2/add*
T0
d
&MobilenetEdgeTPU/expanded_conv_3/inputIdentity'MobilenetEdgeTPU/expanded_conv_2/output*
T0
?
RMobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"          ?   *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_3/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_3/expand/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_3/expand/weights*
dtype0
?
\MobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRMobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_3/expand/weights*
dtype0
?
PMobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normal/mulMul\MobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normal/TruncatedNormalSMobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_3/expand/weights
?
LMobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normalAddPMobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normal/mulQMobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_3/expand/weights
?
/MobilenetEdgeTPU/expanded_conv_3/expand/weights
VariableV2*
shape: ?*
	container *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_3/expand/weights*
dtype0*
shared_name 
?
6MobilenetEdgeTPU/expanded_conv_3/expand/weights/AssignAssign/MobilenetEdgeTPU/expanded_conv_3/expand/weightsLMobilenetEdgeTPU/expanded_conv_3/expand/weights/Initializer/truncated_normal*
use_locking(*
T0*
validate_shape(*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_3/expand/weights
?
4MobilenetEdgeTPU/expanded_conv_3/expand/weights/readIdentity/MobilenetEdgeTPU/expanded_conv_3/expand/weights*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_3/expand/weights*
T0
j
5MobilenetEdgeTPU/expanded_conv_3/expand/dilation_rateConst*
valueB"      *
dtype0
?
.MobilenetEdgeTPU/expanded_conv_3/expand/Conv2DConv2D&MobilenetEdgeTPU/expanded_conv_3/input4MobilenetEdgeTPU/expanded_conv_3/expand/weights/read*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 
?
HMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gamma*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gamma
VariableV2*
shared_name *
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gamma*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gamma/AssignAssign7MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gammaHMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gamma*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gamma/readIdentity7MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gamma*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gamma
?
HMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/beta*
dtype0
?
6MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/beta
VariableV2*
shape:?*
	container *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/beta*
dtype0*
shared_name 
?
=MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/beta/AssignAssign6MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/betaHMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/beta*
use_locking(
?
;MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/beta/readIdentity6MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/beta*
T0*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/beta
?
OMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_mean*
dtype0
?
=MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_mean*
dtype0*
shared_name 
?
DMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_mean/AssignAssign=MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_meanOMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_mean*
use_locking(
?
BMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_mean/readIdentity=MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_mean*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_mean
?
RMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
valueB?*  ??*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_variance
?
AMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_variance
VariableV2*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_variance*
dtype0*
shared_name *
shape:?*
	container 
?
HMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_variance/AssignAssignAMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_varianceRMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_variance*
use_locking(
?
FMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_variance/readIdentityAMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_variance*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_variance*
T0
?
BMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3.MobilenetEdgeTPU/expanded_conv_3/expand/Conv2D<MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/gamma/read;MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/beta/readBMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_mean/readFMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
d
7MobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
,MobilenetEdgeTPU/expanded_conv_3/expand/ReluReluBMobilenetEdgeTPU/expanded_conv_3/expand/BatchNorm/FusedBatchNormV3*
T0
t
1MobilenetEdgeTPU/expanded_conv_3/expansion_outputIdentity,MobilenetEdgeTPU/expanded_conv_3/expand/Relu*
T0
?
SMobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"      ?       *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_3/project/weights
?
RMobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normal/meanConst*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_3/project/weights*
dtype0*
valueB
 *    
?
TMobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_3/project/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normal/shape*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_3/project/weights*
dtype0*
seed2 *
T0*

seed 
?
QMobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_3/project/weights
?
MMobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_3/project/weights
?
0MobilenetEdgeTPU/expanded_conv_3/project/weights
VariableV2*
shape:? *
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_3/project/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_3/project/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_3/project/weightsMMobilenetEdgeTPU/expanded_conv_3/project/weights/Initializer/truncated_normal*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_3/project/weights*
use_locking(
?
5MobilenetEdgeTPU/expanded_conv_3/project/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_3/project/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_3/project/weights
k
6MobilenetEdgeTPU/expanded_conv_3/project/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_3/project/Conv2DConv2D1MobilenetEdgeTPU/expanded_conv_3/expansion_output5MobilenetEdgeTPU/expanded_conv_3/project/weights/read*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
strides
*
T0
?
IMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gamma/Initializer/onesConst*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gamma*
dtype0*
valueB *  ??
?
8MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gamma
VariableV2*
dtype0*
shared_name *
shape: *
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gamma
?
?MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gamma*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gamma*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gamma*
T0
?
IMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/beta/Initializer/zerosConst*
valueB *    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/beta
VariableV2*
shape: *
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/beta
?
<MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB *    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_mean
VariableV2*
shape: *
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
EMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_mean*
use_locking(*
T0
?
CMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_mean*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_mean*
T0
?
SMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB *  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_variance
VariableV2*
shape: *
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_variance*
use_locking(
?
GMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_variance*
T0*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_variance
?
CMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_3/project/Conv2D=MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
e
8MobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
1MobilenetEdgeTPU/expanded_conv_3/project/IdentityIdentityCMobilenetEdgeTPU/expanded_conv_3/project/BatchNorm/FusedBatchNormV3*
T0
?
$MobilenetEdgeTPU/expanded_conv_3/addAddV21MobilenetEdgeTPU/expanded_conv_3/project/Identity&MobilenetEdgeTPU/expanded_conv_3/input*
T0
b
'MobilenetEdgeTPU/expanded_conv_3/outputIdentity$MobilenetEdgeTPU/expanded_conv_3/add*
T0
d
&MobilenetEdgeTPU/expanded_conv_4/inputIdentity'MobilenetEdgeTPU/expanded_conv_3/output*
T0
?
RMobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"          ?   *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_4/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_4/expand/weights
?
SMobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_4/expand/weights*
dtype0
?
\MobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRMobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_4/expand/weights*
dtype0
?
PMobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normal/mulMul\MobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normal/TruncatedNormalSMobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_4/expand/weights
?
LMobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normalAddPMobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normal/mulQMobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_4/expand/weights
?
/MobilenetEdgeTPU/expanded_conv_4/expand/weights
VariableV2*
shape: ?*
	container *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_4/expand/weights*
dtype0*
shared_name 
?
6MobilenetEdgeTPU/expanded_conv_4/expand/weights/AssignAssign/MobilenetEdgeTPU/expanded_conv_4/expand/weightsLMobilenetEdgeTPU/expanded_conv_4/expand/weights/Initializer/truncated_normal*
validate_shape(*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_4/expand/weights*
use_locking(*
T0
?
4MobilenetEdgeTPU/expanded_conv_4/expand/weights/readIdentity/MobilenetEdgeTPU/expanded_conv_4/expand/weights*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_4/expand/weights
j
5MobilenetEdgeTPU/expanded_conv_4/expand/dilation_rateConst*
dtype0*
valueB"      
?
.MobilenetEdgeTPU/expanded_conv_4/expand/Conv2DConv2D&MobilenetEdgeTPU/expanded_conv_4/input4MobilenetEdgeTPU/expanded_conv_4/expand/weights/read*
paddingSAME*
	dilations
*
strides
*
T0*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
HMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gamma*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gamma
VariableV2*
shared_name *
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gamma*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gamma/AssignAssign7MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gammaHMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gamma*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gamma/readIdentity7MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gamma*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gamma
?
HMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/beta*
dtype0
?
6MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/beta
VariableV2*
shape:?*
	container *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/beta*
dtype0*
shared_name 
?
=MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/beta/AssignAssign6MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/betaHMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/beta*
use_locking(
?
;MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/beta/readIdentity6MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/beta*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/beta*
T0
?
OMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_mean*
dtype0
?
=MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_mean
VariableV2*
shared_name *
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_mean*
dtype0
?
DMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_mean/AssignAssign=MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_meanOMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_mean/Initializer/zeros*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
BMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_mean/readIdentity=MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_mean*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_mean*
T0
?
RMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_variance*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_variance
VariableV2*
shared_name *
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_variance*
dtype0
?
HMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_variance/AssignAssignAMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_varianceRMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_variance*
use_locking(*
T0
?
FMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_variance/readIdentityAMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_variance*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_variance
?
BMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3.MobilenetEdgeTPU/expanded_conv_4/expand/Conv2D<MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/gamma/read;MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/beta/readBMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_mean/readFMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
d
7MobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
,MobilenetEdgeTPU/expanded_conv_4/expand/ReluReluBMobilenetEdgeTPU/expanded_conv_4/expand/BatchNorm/FusedBatchNormV3*
T0
t
1MobilenetEdgeTPU/expanded_conv_4/expansion_outputIdentity,MobilenetEdgeTPU/expanded_conv_4/expand/Relu*
T0
?
SMobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?       *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_4/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_4/project/weights
?
TMobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_4/project/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normal/shape*
dtype0*
seed2 *
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_4/project/weights
?
QMobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_4/project/weights
?
MMobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_4/project/weights
?
0MobilenetEdgeTPU/expanded_conv_4/project/weights
VariableV2*
shape:? *
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_4/project/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_4/project/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_4/project/weightsMMobilenetEdgeTPU/expanded_conv_4/project/weights/Initializer/truncated_normal*
use_locking(*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_4/project/weights
?
5MobilenetEdgeTPU/expanded_conv_4/project/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_4/project/weights*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_4/project/weights*
T0
k
6MobilenetEdgeTPU/expanded_conv_4/project/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_4/project/Conv2DConv2D1MobilenetEdgeTPU/expanded_conv_4/expansion_output5MobilenetEdgeTPU/expanded_conv_4/project/weights/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
IMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gamma/Initializer/onesConst*
valueB *  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gamma
VariableV2*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gamma*
dtype0*
shared_name *
shape: *
	container 
?
?MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gamma*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gamma*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gamma
?
IMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/beta/Initializer/zerosConst*
valueB *    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/beta
VariableV2*
shape: *
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/beta
?
<MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB *    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_mean
VariableV2*
shape: *
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
EMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_mean/Initializer/zeros*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
CMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_variance/Initializer/onesConst*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_variance*
dtype0*
valueB *  ??
?
BMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_variance
VariableV2*
shape: *
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_variance*
use_locking(
?
GMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_variance*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_variance*
T0
?
CMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_4/project/Conv2D=MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/moving_variance/read*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0*
T0
e
8MobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
1MobilenetEdgeTPU/expanded_conv_4/project/IdentityIdentityCMobilenetEdgeTPU/expanded_conv_4/project/BatchNorm/FusedBatchNormV3*
T0
?
$MobilenetEdgeTPU/expanded_conv_4/addAddV21MobilenetEdgeTPU/expanded_conv_4/project/Identity&MobilenetEdgeTPU/expanded_conv_4/input*
T0
b
'MobilenetEdgeTPU/expanded_conv_4/outputIdentity$MobilenetEdgeTPU/expanded_conv_4/add*
T0
d
&MobilenetEdgeTPU/expanded_conv_5/inputIdentity'MobilenetEdgeTPU/expanded_conv_4/output*
T0
?
RMobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"             *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_5/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_5/expand/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_5/expand/weights*
dtype0
?
\MobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRMobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normal/shape*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_5/expand/weights*
dtype0*
seed2 *
T0*

seed 
?
PMobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normal/mulMul\MobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normal/TruncatedNormalSMobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_5/expand/weights
?
LMobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normalAddPMobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normal/mulQMobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_5/expand/weights
?
/MobilenetEdgeTPU/expanded_conv_5/expand/weights
VariableV2*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_5/expand/weights*
dtype0*
shared_name *
shape: ?*
	container 
?
6MobilenetEdgeTPU/expanded_conv_5/expand/weights/AssignAssign/MobilenetEdgeTPU/expanded_conv_5/expand/weightsLMobilenetEdgeTPU/expanded_conv_5/expand/weights/Initializer/truncated_normal*
T0*
validate_shape(*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_5/expand/weights*
use_locking(
?
4MobilenetEdgeTPU/expanded_conv_5/expand/weights/readIdentity/MobilenetEdgeTPU/expanded_conv_5/expand/weights*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_5/expand/weights*
T0
j
5MobilenetEdgeTPU/expanded_conv_5/expand/dilation_rateConst*
dtype0*
valueB"      
?
.MobilenetEdgeTPU/expanded_conv_5/expand/Conv2DConv2D&MobilenetEdgeTPU/expanded_conv_5/input4MobilenetEdgeTPU/expanded_conv_5/expand/weights/read*
paddingSAME*
	dilations
*
strides
*
T0*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
HMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gamma*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gamma/AssignAssign7MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gammaHMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gamma*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gamma/readIdentity7MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gamma*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gamma
?
HMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/beta*
dtype0
?
6MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/beta
VariableV2*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/beta*
dtype0*
shared_name *
shape:?*
	container 
?
=MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/beta/AssignAssign6MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/betaHMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/beta*
use_locking(
?
;MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/beta/readIdentity6MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/beta*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/beta*
T0
?
OMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_mean*
dtype0
?
=MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_mean
VariableV2*
dtype0*
shared_name *
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_mean
?
DMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_mean/AssignAssign=MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_meanOMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_mean*
use_locking(
?
BMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_mean/readIdentity=MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_mean*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_mean*
T0
?
RMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_variance*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_variance
VariableV2*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_variance*
dtype0*
shared_name *
shape:?*
	container 
?
HMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_variance/AssignAssignAMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_varianceRMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_variance*
use_locking(*
T0
?
FMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_variance/readIdentityAMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_variance*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_variance
?
BMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3.MobilenetEdgeTPU/expanded_conv_5/expand/Conv2D<MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/gamma/read;MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/beta/readBMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_mean/readFMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
d
7MobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
,MobilenetEdgeTPU/expanded_conv_5/expand/ReluReluBMobilenetEdgeTPU/expanded_conv_5/expand/BatchNorm/FusedBatchNormV3*
T0
t
1MobilenetEdgeTPU/expanded_conv_5/expansion_outputIdentity,MobilenetEdgeTPU/expanded_conv_5/expand/Relu*
T0
?
SMobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"         0   *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_5/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normal/meanConst*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_5/project/weights*
dtype0*
valueB
 *    
?
TMobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_5/project/weights*
dtype0*
valueB
 *?Q?=
?
]MobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normal/shape*
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_5/project/weights*
dtype0*
seed2 
?
QMobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_5/project/weights
?
MMobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_5/project/weights
?
0MobilenetEdgeTPU/expanded_conv_5/project/weights
VariableV2*
shape:?0*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_5/project/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_5/project/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_5/project/weightsMMobilenetEdgeTPU/expanded_conv_5/project/weights/Initializer/truncated_normal*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_5/project/weights*
use_locking(*
T0
?
5MobilenetEdgeTPU/expanded_conv_5/project/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_5/project/weights*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_5/project/weights*
T0
k
6MobilenetEdgeTPU/expanded_conv_5/project/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_5/project/Conv2DConv2D1MobilenetEdgeTPU/expanded_conv_5/expansion_output5MobilenetEdgeTPU/expanded_conv_5/project/weights/read*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 
?
IMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gamma/Initializer/onesConst*
valueB0*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gamma
VariableV2*
shape:0*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gamma*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gamma*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gamma
?
IMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/beta/Initializer/zerosConst*
valueB0*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/beta
VariableV2*
shape:0*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/beta/Initializer/zeros*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/beta*
use_locking(*
T0*
validate_shape(
?
<MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB0*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_mean
VariableV2*
shape:0*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
EMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_mean/Initializer/zeros*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
CMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB0*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_variance
VariableV2*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_variance*
dtype0*
shared_name *
shape:0*
	container 
?
IMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_variance*
use_locking(
?
GMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_variance*
T0*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_variance
?
CMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_5/project/Conv2D=MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/moving_variance/read*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0*
T0
e
8MobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
1MobilenetEdgeTPU/expanded_conv_5/project/IdentityIdentityCMobilenetEdgeTPU/expanded_conv_5/project/BatchNorm/FusedBatchNormV3*
T0
o
'MobilenetEdgeTPU/expanded_conv_5/outputIdentity1MobilenetEdgeTPU/expanded_conv_5/project/Identity*
T0
d
&MobilenetEdgeTPU/expanded_conv_6/inputIdentity'MobilenetEdgeTPU/expanded_conv_5/output*
T0
?
RMobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      0   ?   *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_6/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_6/expand/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_6/expand/weights*
dtype0
?
\MobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRMobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normal/shape*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_6/expand/weights*
dtype0*
seed2 *
T0*

seed 
?
PMobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normal/mulMul\MobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normal/TruncatedNormalSMobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normal/stddev*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_6/expand/weights*
T0
?
LMobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normalAddPMobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normal/mulQMobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_6/expand/weights
?
/MobilenetEdgeTPU/expanded_conv_6/expand/weights
VariableV2*
shared_name *
shape:0?*
	container *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_6/expand/weights*
dtype0
?
6MobilenetEdgeTPU/expanded_conv_6/expand/weights/AssignAssign/MobilenetEdgeTPU/expanded_conv_6/expand/weightsLMobilenetEdgeTPU/expanded_conv_6/expand/weights/Initializer/truncated_normal*
T0*
validate_shape(*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_6/expand/weights*
use_locking(
?
4MobilenetEdgeTPU/expanded_conv_6/expand/weights/readIdentity/MobilenetEdgeTPU/expanded_conv_6/expand/weights*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_6/expand/weights
j
5MobilenetEdgeTPU/expanded_conv_6/expand/dilation_rateConst*
dtype0*
valueB"      
?
.MobilenetEdgeTPU/expanded_conv_6/expand/Conv2DConv2D&MobilenetEdgeTPU/expanded_conv_6/input4MobilenetEdgeTPU/expanded_conv_6/expand/weights/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
HMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gamma*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gamma
VariableV2*
dtype0*
shared_name *
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gamma
?
>MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gamma/AssignAssign7MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gammaHMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gamma*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gamma/readIdentity7MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gamma*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gamma*
T0
?
HMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/beta*
dtype0
?
6MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/beta
VariableV2*
shared_name *
shape:?*
	container *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/beta*
dtype0
?
=MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/beta/AssignAssign6MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/betaHMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/beta/Initializer/zeros*
validate_shape(*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/beta*
use_locking(*
T0
?
;MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/beta/readIdentity6MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/beta*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/beta*
T0
?
OMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_mean*
dtype0
?
=MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_mean
VariableV2*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_mean*
dtype0*
shared_name *
shape:?*
	container 
?
DMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_mean/AssignAssign=MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_meanOMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_mean*
use_locking(*
T0
?
BMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_mean/readIdentity=MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_mean*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_mean
?
RMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
valueB?*  ??*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_variance
?
AMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_variance/AssignAssignAMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_varianceRMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_variance*
use_locking(
?
FMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_variance/readIdentityAMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_variance*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_variance*
T0
?
BMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3.MobilenetEdgeTPU/expanded_conv_6/expand/Conv2D<MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/gamma/read;MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/beta/readBMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_mean/readFMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
d
7MobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
,MobilenetEdgeTPU/expanded_conv_6/expand/ReluReluBMobilenetEdgeTPU/expanded_conv_6/expand/BatchNorm/FusedBatchNormV3*
T0
t
1MobilenetEdgeTPU/expanded_conv_6/expansion_outputIdentity,MobilenetEdgeTPU/expanded_conv_6/expand/Relu*
T0
?
SMobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?   0   *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_6/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_6/project/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_6/project/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normal/shape*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_6/project/weights*
dtype0*
seed2 *
T0*

seed 
?
QMobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normal/stddev*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_6/project/weights*
T0
?
MMobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normal/mean*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_6/project/weights*
T0
?
0MobilenetEdgeTPU/expanded_conv_6/project/weights
VariableV2*
shape:?0*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_6/project/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_6/project/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_6/project/weightsMMobilenetEdgeTPU/expanded_conv_6/project/weights/Initializer/truncated_normal*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_6/project/weights*
use_locking(*
T0
?
5MobilenetEdgeTPU/expanded_conv_6/project/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_6/project/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_6/project/weights
k
6MobilenetEdgeTPU/expanded_conv_6/project/dilation_rateConst*
dtype0*
valueB"      
?
/MobilenetEdgeTPU/expanded_conv_6/project/Conv2DConv2D1MobilenetEdgeTPU/expanded_conv_6/expansion_output5MobilenetEdgeTPU/expanded_conv_6/project/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
IMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gamma/Initializer/onesConst*
valueB0*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gamma
VariableV2*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gamma*
dtype0*
shared_name *
shape:0*
	container 
?
?MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gamma
?
=MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gamma*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gamma
?
IMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/beta/Initializer/zerosConst*
valueB0*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/beta
VariableV2*
shape:0*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/beta/Initializer/zeros*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/beta*
use_locking(*
T0
?
<MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_mean/Initializer/zerosConst*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_mean*
dtype0*
valueB0*    
?
>MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_mean
VariableV2*
dtype0*
shared_name *
shape:0*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_mean
?
EMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_mean*
use_locking(*
T0
?
CMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_mean*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_mean*
T0
?
SMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB0*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_variance
VariableV2*
shape:0*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_variance/Initializer/ones*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_variance*
use_locking(*
T0*
validate_shape(
?
GMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_variance*
T0*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_variance
?
CMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_6/project/Conv2D=MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
e
8MobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
1MobilenetEdgeTPU/expanded_conv_6/project/IdentityIdentityCMobilenetEdgeTPU/expanded_conv_6/project/BatchNorm/FusedBatchNormV3*
T0
?
$MobilenetEdgeTPU/expanded_conv_6/addAddV21MobilenetEdgeTPU/expanded_conv_6/project/Identity&MobilenetEdgeTPU/expanded_conv_6/input*
T0
b
'MobilenetEdgeTPU/expanded_conv_6/outputIdentity$MobilenetEdgeTPU/expanded_conv_6/add*
T0
d
&MobilenetEdgeTPU/expanded_conv_7/inputIdentity'MobilenetEdgeTPU/expanded_conv_6/output*
T0
?
RMobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      0   ?   *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_7/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_7/expand/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_7/expand/weights*
dtype0
?
\MobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRMobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_7/expand/weights*
dtype0
?
PMobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normal/mulMul\MobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normal/TruncatedNormalSMobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_7/expand/weights
?
LMobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normalAddPMobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normal/mulQMobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_7/expand/weights
?
/MobilenetEdgeTPU/expanded_conv_7/expand/weights
VariableV2*
shape:0?*
	container *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_7/expand/weights*
dtype0*
shared_name 
?
6MobilenetEdgeTPU/expanded_conv_7/expand/weights/AssignAssign/MobilenetEdgeTPU/expanded_conv_7/expand/weightsLMobilenetEdgeTPU/expanded_conv_7/expand/weights/Initializer/truncated_normal*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_7/expand/weights*
use_locking(*
T0*
validate_shape(
?
4MobilenetEdgeTPU/expanded_conv_7/expand/weights/readIdentity/MobilenetEdgeTPU/expanded_conv_7/expand/weights*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_7/expand/weights*
T0
j
5MobilenetEdgeTPU/expanded_conv_7/expand/dilation_rateConst*
dtype0*
valueB"      
?
.MobilenetEdgeTPU/expanded_conv_7/expand/Conv2DConv2D&MobilenetEdgeTPU/expanded_conv_7/input4MobilenetEdgeTPU/expanded_conv_7/expand/weights/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
HMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gamma/Initializer/onesConst*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gamma*
dtype0*
valueB?*  ??
?
7MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gamma/AssignAssign7MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gammaHMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gamma*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gamma/readIdentity7MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gamma*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gamma*
T0
?
HMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/beta*
dtype0
?
6MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/beta
VariableV2*
shared_name *
shape:?*
	container *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/beta*
dtype0
?
=MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/beta/AssignAssign6MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/betaHMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/beta*
use_locking(
?
;MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/beta/readIdentity6MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/beta*
T0*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/beta
?
OMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_mean
?
=MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_mean
VariableV2*
shared_name *
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_mean*
dtype0
?
DMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_mean/AssignAssign=MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_meanOMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_mean*
use_locking(
?
BMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_mean/readIdentity=MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_mean*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_mean*
T0
?
RMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_variance*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_variance/AssignAssignAMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_varianceRMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_variance*
use_locking(*
T0
?
FMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_variance/readIdentityAMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_variance*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_variance
?
BMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3.MobilenetEdgeTPU/expanded_conv_7/expand/Conv2D<MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/gamma/read;MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/beta/readBMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_mean/readFMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
d
7MobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
,MobilenetEdgeTPU/expanded_conv_7/expand/ReluReluBMobilenetEdgeTPU/expanded_conv_7/expand/BatchNorm/FusedBatchNormV3*
T0
t
1MobilenetEdgeTPU/expanded_conv_7/expansion_outputIdentity,MobilenetEdgeTPU/expanded_conv_7/expand/Relu*
T0
?
SMobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?   0   *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_7/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_7/project/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_7/project/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_7/project/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normal/stddev*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_7/project/weights*
T0
?
MMobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_7/project/weights
?
0MobilenetEdgeTPU/expanded_conv_7/project/weights
VariableV2*
shared_name *
shape:?0*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_7/project/weights*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_7/project/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_7/project/weightsMMobilenetEdgeTPU/expanded_conv_7/project/weights/Initializer/truncated_normal*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_7/project/weights*
use_locking(*
T0*
validate_shape(
?
5MobilenetEdgeTPU/expanded_conv_7/project/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_7/project/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_7/project/weights
k
6MobilenetEdgeTPU/expanded_conv_7/project/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_7/project/Conv2DConv2D1MobilenetEdgeTPU/expanded_conv_7/expansion_output5MobilenetEdgeTPU/expanded_conv_7/project/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
IMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gamma/Initializer/onesConst*
valueB0*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gamma
VariableV2*
dtype0*
shared_name *
shape:0*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gamma
?
?MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gamma*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gamma*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gamma*
T0
?
IMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/beta/Initializer/zerosConst*
valueB0*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/beta
VariableV2*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/beta*
dtype0*
shared_name *
shape:0*
	container 
?
>MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/beta*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB0*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_mean
VariableV2*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_mean*
dtype0*
shared_name *
shape:0*
	container 
?
EMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_mean/Initializer/zeros*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
CMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB0*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_variance
VariableV2*
shape:0*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_variance*
use_locking(*
T0
?
GMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_variance*
T0*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_variance
?
CMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_7/project/Conv2D=MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
e
8MobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
1MobilenetEdgeTPU/expanded_conv_7/project/IdentityIdentityCMobilenetEdgeTPU/expanded_conv_7/project/BatchNorm/FusedBatchNormV3*
T0
?
$MobilenetEdgeTPU/expanded_conv_7/addAddV21MobilenetEdgeTPU/expanded_conv_7/project/Identity&MobilenetEdgeTPU/expanded_conv_7/input*
T0
b
'MobilenetEdgeTPU/expanded_conv_7/outputIdentity$MobilenetEdgeTPU/expanded_conv_7/add*
T0
d
&MobilenetEdgeTPU/expanded_conv_8/inputIdentity'MobilenetEdgeTPU/expanded_conv_7/output*
T0
?
RMobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      0   ?   *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_8/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_8/expand/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normal/stddevConst*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_8/expand/weights*
dtype0*
valueB
 *?Q?=
?
\MobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRMobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normal/shape*
dtype0*
seed2 *
T0*

seed *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_8/expand/weights
?
PMobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normal/mulMul\MobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normal/TruncatedNormalSMobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_8/expand/weights
?
LMobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normalAddPMobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normal/mulQMobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_8/expand/weights
?
/MobilenetEdgeTPU/expanded_conv_8/expand/weights
VariableV2*
shared_name *
shape:0?*
	container *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_8/expand/weights*
dtype0
?
6MobilenetEdgeTPU/expanded_conv_8/expand/weights/AssignAssign/MobilenetEdgeTPU/expanded_conv_8/expand/weightsLMobilenetEdgeTPU/expanded_conv_8/expand/weights/Initializer/truncated_normal*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_8/expand/weights*
use_locking(*
T0*
validate_shape(
?
4MobilenetEdgeTPU/expanded_conv_8/expand/weights/readIdentity/MobilenetEdgeTPU/expanded_conv_8/expand/weights*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_8/expand/weights*
T0
j
5MobilenetEdgeTPU/expanded_conv_8/expand/dilation_rateConst*
valueB"      *
dtype0
?
.MobilenetEdgeTPU/expanded_conv_8/expand/Conv2DConv2D&MobilenetEdgeTPU/expanded_conv_8/input4MobilenetEdgeTPU/expanded_conv_8/expand/weights/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
HMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gamma*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gamma/AssignAssign7MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gammaHMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gamma*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gamma/readIdentity7MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gamma*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gamma*
T0
?
HMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/beta/Initializer/zerosConst*
dtype0*
valueB?*    *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/beta
?
6MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/beta
VariableV2*
shape:?*
	container *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/beta*
dtype0*
shared_name 
?
=MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/beta/AssignAssign6MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/betaHMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/beta*
use_locking(
?
;MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/beta/readIdentity6MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/beta*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/beta*
T0
?
OMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_mean/Initializer/zerosConst*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_mean*
dtype0*
valueB?*    
?
=MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_mean*
dtype0*
shared_name 
?
DMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_mean/AssignAssign=MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_meanOMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_mean*
use_locking(
?
BMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_mean/readIdentity=MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_mean*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_mean*
T0
?
RMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_variance*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_variance
VariableV2*
shared_name *
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_variance*
dtype0
?
HMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_variance/AssignAssignAMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_varianceRMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_variance/readIdentityAMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_variance*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_variance*
T0
?
BMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3.MobilenetEdgeTPU/expanded_conv_8/expand/Conv2D<MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/gamma/read;MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/beta/readBMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_mean/readFMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
d
7MobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
,MobilenetEdgeTPU/expanded_conv_8/expand/ReluReluBMobilenetEdgeTPU/expanded_conv_8/expand/BatchNorm/FusedBatchNormV3*
T0
t
1MobilenetEdgeTPU/expanded_conv_8/expansion_outputIdentity,MobilenetEdgeTPU/expanded_conv_8/expand/Relu*
T0
?
SMobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?   0   *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_8/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_8/project/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_8/project/weights*
dtype0*
valueB
 *?Q?=
?
]MobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normal/shape*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_8/project/weights*
dtype0*
seed2 *
T0*

seed 
?
QMobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_8/project/weights
?
MMobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normal/mean*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_8/project/weights*
T0
?
0MobilenetEdgeTPU/expanded_conv_8/project/weights
VariableV2*
shape:?0*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_8/project/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_8/project/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_8/project/weightsMMobilenetEdgeTPU/expanded_conv_8/project/weights/Initializer/truncated_normal*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_8/project/weights*
use_locking(
?
5MobilenetEdgeTPU/expanded_conv_8/project/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_8/project/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_8/project/weights
k
6MobilenetEdgeTPU/expanded_conv_8/project/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_8/project/Conv2DConv2D1MobilenetEdgeTPU/expanded_conv_8/expansion_output5MobilenetEdgeTPU/expanded_conv_8/project/weights/read*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 
?
IMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gamma/Initializer/onesConst*
dtype0*
valueB0*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gamma
?
8MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gamma
VariableV2*
shape:0*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gamma/Initializer/ones*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gamma*
use_locking(*
T0
?
=MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gamma*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gamma*
T0
?
IMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/beta/Initializer/zerosConst*
dtype0*
valueB0*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/beta
?
7MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/beta
VariableV2*
shape:0*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/beta/Initializer/zeros*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/beta*
use_locking(*
T0
?
<MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB0*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_mean
VariableV2*
shape:0*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
EMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_mean/Initializer/zeros*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
CMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
valueB0*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_variance
?
BMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_variance
VariableV2*
shape:0*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_variance*
use_locking(
?
GMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_variance*
T0*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_variance
?
CMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_8/project/Conv2D=MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
e
8MobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
1MobilenetEdgeTPU/expanded_conv_8/project/IdentityIdentityCMobilenetEdgeTPU/expanded_conv_8/project/BatchNorm/FusedBatchNormV3*
T0
?
$MobilenetEdgeTPU/expanded_conv_8/addAddV21MobilenetEdgeTPU/expanded_conv_8/project/Identity&MobilenetEdgeTPU/expanded_conv_8/input*
T0
b
'MobilenetEdgeTPU/expanded_conv_8/outputIdentity$MobilenetEdgeTPU/expanded_conv_8/add*
T0
d
&MobilenetEdgeTPU/expanded_conv_9/inputIdentity'MobilenetEdgeTPU/expanded_conv_8/output*
T0
?
RMobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      0   ?  *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_9/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_9/expand/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normal/stddevConst*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_9/expand/weights*
dtype0*
valueB
 *?Q?=
?
\MobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRMobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normal/shape*
dtype0*
seed2 *
T0*

seed *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_9/expand/weights
?
PMobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normal/mulMul\MobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normal/TruncatedNormalSMobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_9/expand/weights
?
LMobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normalAddPMobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normal/mulQMobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_9/expand/weights
?
/MobilenetEdgeTPU/expanded_conv_9/expand/weights
VariableV2*
shape:0?*
	container *B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_9/expand/weights*
dtype0*
shared_name 
?
6MobilenetEdgeTPU/expanded_conv_9/expand/weights/AssignAssign/MobilenetEdgeTPU/expanded_conv_9/expand/weightsLMobilenetEdgeTPU/expanded_conv_9/expand/weights/Initializer/truncated_normal*
validate_shape(*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_9/expand/weights*
use_locking(*
T0
?
4MobilenetEdgeTPU/expanded_conv_9/expand/weights/readIdentity/MobilenetEdgeTPU/expanded_conv_9/expand/weights*
T0*B
_class8
64loc:@MobilenetEdgeTPU/expanded_conv_9/expand/weights
j
5MobilenetEdgeTPU/expanded_conv_9/expand/dilation_rateConst*
valueB"      *
dtype0
?
.MobilenetEdgeTPU/expanded_conv_9/expand/Conv2DConv2D&MobilenetEdgeTPU/expanded_conv_9/input4MobilenetEdgeTPU/expanded_conv_9/expand/weights/read*
strides
*
T0*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
HMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gamma*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gamma/AssignAssign7MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gammaHMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gamma*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gamma/readIdentity7MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gamma*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gamma
?
HMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/beta/Initializer/zerosConst*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/beta*
dtype0*
valueB?*    
?
6MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/beta
VariableV2*
dtype0*
shared_name *
shape:?*
	container *I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/beta
?
=MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/beta/AssignAssign6MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/betaHMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/beta*
use_locking(
?
;MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/beta/readIdentity6MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/beta*I
_class?
=;loc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/beta*
T0
?
OMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_mean*
dtype0
?
=MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_mean
VariableV2*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_mean*
dtype0*
shared_name *
shape:?*
	container 
?
DMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_mean/AssignAssign=MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_meanOMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_mean*
use_locking(
?
BMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_mean/readIdentity=MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_mean*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_mean
?
RMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_variance*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_variance
VariableV2*
dtype0*
shared_name *
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_variance
?
HMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_variance/AssignAssignAMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_varianceRMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_variance*
use_locking(
?
FMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_variance/readIdentityAMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_variance*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_variance
?
BMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3.MobilenetEdgeTPU/expanded_conv_9/expand/Conv2D<MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/gamma/read;MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/beta/readBMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_mean/readFMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/moving_variance/read*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0*
T0
d
7MobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
,MobilenetEdgeTPU/expanded_conv_9/expand/ReluReluBMobilenetEdgeTPU/expanded_conv_9/expand/BatchNorm/FusedBatchNormV3*
T0
t
1MobilenetEdgeTPU/expanded_conv_9/expansion_outputIdentity,MobilenetEdgeTPU/expanded_conv_9/expand/Relu*
T0
?
_MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?     *O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights*
dtype0
?
`MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights*
dtype0
?
iMobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal_MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normal/shape*O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights*
dtype0*
seed2 *
T0*

seed 
?
]MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuliMobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormal`MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights*
T0
?
YMobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normalAdd]MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normal/mul^MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normal/mean*O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights*
T0
?
<MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights
VariableV2*
shared_name *
shape:?*
	container *O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights*
dtype0
?
CMobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/AssignAssign<MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weightsYMobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/Initializer/truncated_normal*
validate_shape(*O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights*
use_locking(*
T0
?
AMobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/readIdentity<MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights*
T0*O
_classE
CAloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights
w
:MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise/ShapeConst*
dtype0*%
valueB"      ?     
w
BMobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise/dilation_rateConst*
dtype0*
valueB"      
?
4MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwiseDepthwiseConv2dNative1MobilenetEdgeTPU/expanded_conv_9/expansion_outputAMobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise_weights/read*
strides
*
T0*
paddingSAME*
data_formatNHWC*
	dilations

?
KMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gamma*
dtype0
?
:MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gamma
VariableV2*
shape:?*
	container *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gamma*
dtype0*
shared_name 
?
AMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gamma/AssignAssign:MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gammaKMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gamma*
use_locking(
?
?MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gamma/readIdentity:MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gamma*
T0*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gamma
?
KMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/beta*
dtype0
?
9MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/beta
VariableV2*
shape:?*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/beta*
dtype0*
shared_name 
?
@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/beta/AssignAssign9MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/betaKMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/beta/Initializer/zeros*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/beta*
use_locking(*
T0
?
>MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/beta/readIdentity9MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/beta*
T0*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/beta
?
RMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *S
_classI
GEloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_mean*
dtype0
?
@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *S
_classI
GEloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_mean*
dtype0*
shared_name 
?
GMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_mean/AssignAssign@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_meanRMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_mean/Initializer/zeros*S
_classI
GEloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
EMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_mean/readIdentity@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_mean*
T0*S
_classI
GEloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_mean
?
UMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
valueB?*  ??*W
_classM
KIloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_variance
?
DMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *W
_classM
KIloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_variance*
dtype0*
shared_name 
?
KMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_variance/AssignAssignDMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_varianceUMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*W
_classM
KIloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_variance*
use_locking(*
T0
?
IMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_variance/readIdentityDMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_variance*
T0*W
_classM
KIloc:@MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_variance
?
EMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV34MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise?MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/gamma/read>MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/beta/readEMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_mean/readIMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/moving_variance/read*
epsilon%o?:*
U0*
T0*
data_formatNHWC*
is_training( 
g
:MobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
/MobilenetEdgeTPU/expanded_conv_9/depthwise/ReluReluEMobilenetEdgeTPU/expanded_conv_9/depthwise/BatchNorm/FusedBatchNormV3*
T0
w
1MobilenetEdgeTPU/expanded_conv_9/depthwise_outputIdentity/MobilenetEdgeTPU/expanded_conv_9/depthwise/Relu*
T0
?
SMobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?  `   *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_9/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_9/project/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_9/project/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normal/shape*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_9/project/weights*
dtype0*
seed2 *
T0*

seed 
?
QMobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_9/project/weights
?
MMobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normal/mean*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_9/project/weights*
T0
?
0MobilenetEdgeTPU/expanded_conv_9/project/weights
VariableV2*
shape:?`*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_9/project/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_9/project/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_9/project/weightsMMobilenetEdgeTPU/expanded_conv_9/project/weights/Initializer/truncated_normal*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_9/project/weights*
use_locking(
?
5MobilenetEdgeTPU/expanded_conv_9/project/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_9/project/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_9/project/weights
k
6MobilenetEdgeTPU/expanded_conv_9/project/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_9/project/Conv2DConv2D1MobilenetEdgeTPU/expanded_conv_9/depthwise_output5MobilenetEdgeTPU/expanded_conv_9/project/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
IMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gamma/Initializer/onesConst*
valueB`*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gamma
VariableV2*
shape:`*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gamma/Initializer/ones*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gamma*
use_locking(*
T0
?
=MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gamma*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gamma*
T0
?
IMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/beta/Initializer/zerosConst*
valueB`*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/beta
VariableV2*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/beta*
dtype0*
shared_name *
shape:`*
	container 
?
>MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/beta/Initializer/zeros*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/beta*
use_locking(*
T0*
validate_shape(
?
<MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/beta*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/beta*
T0
?
PMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB`*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_mean
?
>MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_mean
VariableV2*
shared_name *
shape:`*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_mean*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_mean
?
CMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB`*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_variance
VariableV2*
shape:`*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_variance*
use_locking(
?
GMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_variance*
T0*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_variance
?
CMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_9/project/Conv2D=MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
e
8MobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
1MobilenetEdgeTPU/expanded_conv_9/project/IdentityIdentityCMobilenetEdgeTPU/expanded_conv_9/project/BatchNorm/FusedBatchNormV3*
T0
o
'MobilenetEdgeTPU/expanded_conv_9/outputIdentity1MobilenetEdgeTPU/expanded_conv_9/project/Identity*
T0
e
'MobilenetEdgeTPU/expanded_conv_10/inputIdentity'MobilenetEdgeTPU/expanded_conv_9/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      `   ?  *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_10/expand/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normal/meanConst*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_10/expand/weights*
dtype0*
valueB
 *    
?
TMobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_10/expand/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_10/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normal/stddev*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_10/expand/weights*
T0
?
MMobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_10/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_10/expand/weights
VariableV2*
shape:`?*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_10/expand/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_10/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_10/expand/weightsMMobilenetEdgeTPU/expanded_conv_10/expand/weights/Initializer/truncated_normal*
use_locking(*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_10/expand/weights
?
5MobilenetEdgeTPU/expanded_conv_10/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_10/expand/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_10/expand/weights
k
6MobilenetEdgeTPU/expanded_conv_10/expand/dilation_rateConst*
dtype0*
valueB"      
?
/MobilenetEdgeTPU/expanded_conv_10/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_10/input5MobilenetEdgeTPU/expanded_conv_10/expand/weights/read*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 
?
IMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gamma/Initializer/onesConst*
dtype0*
valueB?*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gamma
?
8MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gamma/Initializer/ones*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gamma*
use_locking(*
T0
?
=MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gamma*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gamma
?
IMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/beta
VariableV2*
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/beta/Initializer/zeros*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/beta*
use_locking(*
T0
?
<MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/beta*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/beta*
T0
?
PMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_mean
VariableV2*
dtype0*
shared_name *
shape:?*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_mean
?
EMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_mean*
use_locking(*
T0
?
CMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
valueB?*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_variance
?
BMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_variance*
use_locking(*
T0
?
GMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_variance*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_variance*
T0
?
CMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_10/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/moving_variance/read*
epsilon%o?:*
U0*
T0*
data_formatNHWC*
is_training( 
e
8MobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
-MobilenetEdgeTPU/expanded_conv_10/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_10/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_10/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_10/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?     *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights*
dtype0
?
_MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights*
dtype0
?
aMobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights*
dtype0
?
jMobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normal/shape*
T0*

seed *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights*
dtype0*
seed2 
?
^MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights
?
ZMobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normal/mean*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights
?
=MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights
VariableV2*
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights*
dtype0*
shared_name 
?
DMobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/Initializer/truncated_normal*
T0*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights*
use_locking(
?
BMobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights
x
;MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise/ShapeConst*%
valueB"      ?     *
dtype0
x
CMobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0
?
5MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_10/expansion_outputBMobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise_weights/read*
	dilations
*
strides
*
T0*
paddingSAME*
data_formatNHWC
?
LMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gamma*
dtype0
?
;MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gamma
VariableV2*
shared_name *
shape:?*
	container *N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gamma*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gamma/Initializer/ones*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gamma*
use_locking(*
T0*
validate_shape(
?
@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gamma*
T0*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gamma
?
LMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/beta*
dtype0
?
:MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/beta
VariableV2*
shared_name *
shape:?*
	container *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/beta*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/beta*
use_locking(
?
?MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/beta*
T0*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/beta
?
SMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_mean*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_mean
VariableV2*
shared_name *
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_mean*
dtype0
?
HMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_mean*
use_locking(*
T0
?
FMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_mean*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_mean*
T0
?
VMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_variance*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_variance*
dtype0*
shared_name 
?
LMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_variance*
use_locking(
?
JMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_variance*
T0*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
h
;MobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
0MobilenetEdgeTPU/expanded_conv_10/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_10/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_10/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_10/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?  `   *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_10/project/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_10/project/weights*
dtype0
?
UMobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_10/project/weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_10/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normal/stddev*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_10/project/weights*
T0
?
NMobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normal/mean*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_10/project/weights*
T0
?
1MobilenetEdgeTPU/expanded_conv_10/project/weights
VariableV2*
shape:?`*
	container *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_10/project/weights*
dtype0*
shared_name 
?
8MobilenetEdgeTPU/expanded_conv_10/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_10/project/weightsNMobilenetEdgeTPU/expanded_conv_10/project/weights/Initializer/truncated_normal*
T0*
validate_shape(*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_10/project/weights*
use_locking(
?
6MobilenetEdgeTPU/expanded_conv_10/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_10/project/weights*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_10/project/weights
l
7MobilenetEdgeTPU/expanded_conv_10/project/dilation_rateConst*
valueB"      *
dtype0
?
0MobilenetEdgeTPU/expanded_conv_10/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_10/depthwise_output6MobilenetEdgeTPU/expanded_conv_10/project/weights/read*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC
?
JMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gamma/Initializer/onesConst*
valueB`*  ??*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gamma*
dtype0
?
9MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gamma
VariableV2*
shape:`*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gamma*
dtype0*
shared_name 
?
@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gamma/Initializer/ones*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gamma*
use_locking(*
T0
?
>MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gamma*
T0*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gamma
?
JMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/beta/Initializer/zerosConst*
valueB`*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/beta
VariableV2*
shape:`*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/beta*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/beta*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/beta*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/beta
?
QMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB`*    *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_mean*
dtype0
?
?MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_mean
VariableV2*
shape:`*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
FMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_mean/Initializer/zeros*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
DMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_mean*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_mean*
T0
?
TMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB`*  ??*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_variance*
dtype0
?
CMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_variance
VariableV2*
dtype0*
shared_name *
shape:`*
	container *V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_variance
?
JMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_variance*
use_locking(
?
HMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_variance*
T0*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_variance
?
DMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_10/project/Conv2D>MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/moving_variance/read*
epsilon%o?:*
U0*
T0*
data_formatNHWC*
is_training( 
f
9MobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
2MobilenetEdgeTPU/expanded_conv_10/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_10/project/BatchNorm/FusedBatchNormV3*
T0
?
%MobilenetEdgeTPU/expanded_conv_10/addAddV22MobilenetEdgeTPU/expanded_conv_10/project/Identity'MobilenetEdgeTPU/expanded_conv_10/input*
T0
d
(MobilenetEdgeTPU/expanded_conv_10/outputIdentity%MobilenetEdgeTPU/expanded_conv_10/add*
T0
f
'MobilenetEdgeTPU/expanded_conv_11/inputIdentity(MobilenetEdgeTPU/expanded_conv_10/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_11/expand/weights*
dtype0*%
valueB"      `   ?  
?
RMobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_11/expand/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_11/expand/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_11/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_11/expand/weights
?
MMobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_11/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_11/expand/weights
VariableV2*
shape:`?*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_11/expand/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_11/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_11/expand/weightsMMobilenetEdgeTPU/expanded_conv_11/expand/weights/Initializer/truncated_normal*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_11/expand/weights*
use_locking(
?
5MobilenetEdgeTPU/expanded_conv_11/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_11/expand/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_11/expand/weights
k
6MobilenetEdgeTPU/expanded_conv_11/expand/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_11/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_11/input5MobilenetEdgeTPU/expanded_conv_11/expand/weights/read*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC
?
IMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gamma/Initializer/ones*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gamma*
use_locking(*
T0
?
=MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gamma*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gamma*
T0
?
IMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/beta/Initializer/zerosConst*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/beta*
dtype0*
valueB?*    
?
7MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/beta
VariableV2*
dtype0*
shared_name *
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/beta
?
>MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/beta*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/beta*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/beta*
T0
?
PMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_mean*
dtype0*
shared_name 
?
EMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_mean*
use_locking(
?
CMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_variance*
use_locking(*
T0
?
GMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_variance*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_variance*
T0
?
CMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_11/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
e
8MobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
-MobilenetEdgeTPU/expanded_conv_11/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_11/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_11/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_11/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights*
dtype0*%
valueB"      ?     
?
_MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights*
dtype0
?
aMobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights*
dtype0
?
jMobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normal/shape*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights*
dtype0*
seed2 *
T0*

seed 
?
^MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights*
T0
?
ZMobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normal/mean*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights*
T0
?
=MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights
VariableV2*
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights*
dtype0*
shared_name 
?
DMobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/Initializer/truncated_normal*
T0*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights*
use_locking(
?
BMobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights
x
;MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise/ShapeConst*%
valueB"      ?     *
dtype0
x
CMobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0
?
5MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_11/expansion_outputBMobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise_weights/read*
strides
*
T0*
paddingSAME*
data_formatNHWC*
	dilations

?
LMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gamma*
dtype0
?
;MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gamma
VariableV2*
shared_name *
shape:?*
	container *N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gamma*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gamma/Initializer/ones*
validate_shape(*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gamma*
use_locking(*
T0
?
@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gamma*
T0*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gamma
?
LMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/beta*
dtype0
?
:MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/beta
VariableV2*
shared_name *
shape:?*
	container *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/beta*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/beta*
use_locking(
?
?MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/beta*
T0*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/beta
?
SMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_mean*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_mean*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_mean*
use_locking(
?
FMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_mean*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_mean
?
VMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_variance*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_variance*
dtype0*
shared_name 
?
LMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_variance/Initializer/ones*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_variance*
use_locking(*
T0*
validate_shape(
?
JMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_variance*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_variance*
T0
?
FMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
h
;MobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
0MobilenetEdgeTPU/expanded_conv_11/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_11/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_11/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_11/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?  `   *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_11/project/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_11/project/weights*
dtype0
?
UMobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_11/project/weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normal/shape*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_11/project/weights*
dtype0*
seed2 *
T0*

seed 
?
RMobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_11/project/weights
?
NMobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normal/mean*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_11/project/weights*
T0
?
1MobilenetEdgeTPU/expanded_conv_11/project/weights
VariableV2*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_11/project/weights*
dtype0*
shared_name *
shape:?`*
	container 
?
8MobilenetEdgeTPU/expanded_conv_11/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_11/project/weightsNMobilenetEdgeTPU/expanded_conv_11/project/weights/Initializer/truncated_normal*
T0*
validate_shape(*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_11/project/weights*
use_locking(
?
6MobilenetEdgeTPU/expanded_conv_11/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_11/project/weights*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_11/project/weights
l
7MobilenetEdgeTPU/expanded_conv_11/project/dilation_rateConst*
valueB"      *
dtype0
?
0MobilenetEdgeTPU/expanded_conv_11/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_11/depthwise_output6MobilenetEdgeTPU/expanded_conv_11/project/weights/read*
strides
*
T0*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
JMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gamma/Initializer/onesConst*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gamma*
dtype0*
valueB`*  ??
?
9MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gamma
VariableV2*
shared_name *
shape:`*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gamma*
dtype0
?
@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gamma
?
>MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gamma*
T0*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gamma
?
JMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/beta/Initializer/zerosConst*
valueB`*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/beta
VariableV2*
shape:`*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/beta*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/beta*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/beta*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/beta
?
QMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB`*    *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_mean*
dtype0
?
?MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_mean
VariableV2*
shared_name *
shape:`*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_mean*
dtype0
?
FMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_mean/Initializer/zeros*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
DMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_mean*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_mean*
T0
?
TMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB`*  ??*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_variance*
dtype0
?
CMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_variance
VariableV2*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_variance*
dtype0*
shared_name *
shape:`*
	container 
?
JMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_variance*
use_locking(
?
HMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_variance*
T0*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_variance
?
DMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_11/project/Conv2D>MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/moving_variance/read*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0*
T0
f
9MobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
2MobilenetEdgeTPU/expanded_conv_11/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_11/project/BatchNorm/FusedBatchNormV3*
T0
?
%MobilenetEdgeTPU/expanded_conv_11/addAddV22MobilenetEdgeTPU/expanded_conv_11/project/Identity'MobilenetEdgeTPU/expanded_conv_11/input*
T0
d
(MobilenetEdgeTPU/expanded_conv_11/outputIdentity%MobilenetEdgeTPU/expanded_conv_11/add*
T0
f
'MobilenetEdgeTPU/expanded_conv_12/inputIdentity(MobilenetEdgeTPU/expanded_conv_11/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      `   ?  *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_12/expand/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_12/expand/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_12/expand/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normal/shape*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_12/expand/weights*
dtype0*
seed2 *
T0*

seed 
?
QMobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_12/expand/weights
?
MMobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_12/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_12/expand/weights
VariableV2*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_12/expand/weights*
dtype0*
shared_name *
shape:`?*
	container 
?
7MobilenetEdgeTPU/expanded_conv_12/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_12/expand/weightsMMobilenetEdgeTPU/expanded_conv_12/expand/weights/Initializer/truncated_normal*
use_locking(*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_12/expand/weights
?
5MobilenetEdgeTPU/expanded_conv_12/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_12/expand/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_12/expand/weights
k
6MobilenetEdgeTPU/expanded_conv_12/expand/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_12/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_12/input5MobilenetEdgeTPU/expanded_conv_12/expand/weights/read*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
strides
*
T0
?
IMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gamma/Initializer/ones*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gamma*
use_locking(*
T0
?
=MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gamma*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gamma*
T0
?
IMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/beta
VariableV2*
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/beta/Initializer/zeros*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/beta*
use_locking(*
T0
?
<MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/beta*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/beta*
T0
?
PMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_mean*
dtype0*
shared_name 
?
EMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_mean/Initializer/zeros*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
CMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_variance
VariableV2*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_variance*
dtype0*
shared_name *
shape:?*
	container 
?
IMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_variance*
use_locking(*
T0
?
GMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_variance*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_variance*
T0
?
CMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_12/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
e
8MobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
-MobilenetEdgeTPU/expanded_conv_12/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_12/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_12/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_12/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights*
dtype0*%
valueB"      ?     
?
_MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights*
dtype0*
valueB
 *    
?
aMobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights*
dtype0
?
jMobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normal/shape*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights*
dtype0*
seed2 *
T0*

seed 
?
^MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights
?
ZMobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normal/mean*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights*
T0
?
=MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights
VariableV2*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights*
dtype0*
shared_name *
shape:?*
	container 
?
DMobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/Initializer/truncated_normal*
T0*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights*
use_locking(
?
BMobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights*
T0
x
;MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise/ShapeConst*%
valueB"      ?     *
dtype0
x
CMobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0
?
5MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_12/expansion_outputBMobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise_weights/read*
strides
*
T0*
paddingSAME*
data_formatNHWC*
	dilations

?
LMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gamma/Initializer/onesConst*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gamma*
dtype0*
valueB?*  ??
?
;MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gamma
VariableV2*
shared_name *
shape:?*
	container *N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gamma*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gamma/Initializer/ones*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gamma*
use_locking(*
T0*
validate_shape(
?
@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gamma*
T0*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gamma
?
LMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/beta*
dtype0
?
:MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/beta
VariableV2*
dtype0*
shared_name *
shape:?*
	container *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/beta
?
AMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/beta*
use_locking(
?
?MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/beta*
T0*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/beta
?
SMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_mean*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_mean*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_mean*
use_locking(*
T0
?
FMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_mean*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_mean*
T0
?
VMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_variance*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_variance
VariableV2*
dtype0*
shared_name *
shape:?*
	container *X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_variance
?
LMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*
validate_shape(*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_variance
?
JMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_variance*
T0*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
h
;MobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
0MobilenetEdgeTPU/expanded_conv_12/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_12/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_12/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_12/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?  `   *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_12/project/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_12/project/weights*
dtype0
?
UMobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_12/project/weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_12/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_12/project/weights
?
NMobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normal/mean*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_12/project/weights
?
1MobilenetEdgeTPU/expanded_conv_12/project/weights
VariableV2*
shape:?`*
	container *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_12/project/weights*
dtype0*
shared_name 
?
8MobilenetEdgeTPU/expanded_conv_12/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_12/project/weightsNMobilenetEdgeTPU/expanded_conv_12/project/weights/Initializer/truncated_normal*
T0*
validate_shape(*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_12/project/weights*
use_locking(
?
6MobilenetEdgeTPU/expanded_conv_12/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_12/project/weights*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_12/project/weights
l
7MobilenetEdgeTPU/expanded_conv_12/project/dilation_rateConst*
dtype0*
valueB"      
?
0MobilenetEdgeTPU/expanded_conv_12/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_12/depthwise_output6MobilenetEdgeTPU/expanded_conv_12/project/weights/read*
strides
*
T0*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
JMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gamma/Initializer/onesConst*
valueB`*  ??*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gamma*
dtype0
?
9MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gamma
VariableV2*
shape:`*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gamma*
dtype0*
shared_name 
?
@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gamma/Initializer/ones*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gamma*
use_locking(*
T0
?
>MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gamma*
T0*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gamma
?
JMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/beta/Initializer/zerosConst*
valueB`*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/beta
VariableV2*
shape:`*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/beta*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/beta*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/beta*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/beta*
T0
?
QMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB`*    *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_mean*
dtype0
?
?MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_mean
VariableV2*
shape:`*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
FMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_mean/Initializer/zeros*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
DMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_mean*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_mean*
T0
?
TMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB`*  ??*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_variance*
dtype0
?
CMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_variance
VariableV2*
shared_name *
shape:`*
	container *V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_variance*
dtype0
?
JMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*
validate_shape(*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_variance
?
HMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_variance*
T0*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_variance
?
DMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_12/project/Conv2D>MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/moving_variance/read*
epsilon%o?:*
U0*
T0*
data_formatNHWC*
is_training( 
f
9MobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
2MobilenetEdgeTPU/expanded_conv_12/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_12/project/BatchNorm/FusedBatchNormV3*
T0
?
%MobilenetEdgeTPU/expanded_conv_12/addAddV22MobilenetEdgeTPU/expanded_conv_12/project/Identity'MobilenetEdgeTPU/expanded_conv_12/input*
T0
d
(MobilenetEdgeTPU/expanded_conv_12/outputIdentity%MobilenetEdgeTPU/expanded_conv_12/add*
T0
f
'MobilenetEdgeTPU/expanded_conv_13/inputIdentity(MobilenetEdgeTPU/expanded_conv_12/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      `      *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_13/expand/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_13/expand/weights
?
TMobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_13/expand/weights
?
]MobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_13/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_13/expand/weights
?
MMobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_13/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_13/expand/weights
VariableV2*
shape:`?*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_13/expand/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_13/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_13/expand/weightsMMobilenetEdgeTPU/expanded_conv_13/expand/weights/Initializer/truncated_normal*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_13/expand/weights*
use_locking(
?
5MobilenetEdgeTPU/expanded_conv_13/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_13/expand/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_13/expand/weights
k
6MobilenetEdgeTPU/expanded_conv_13/expand/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_13/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_13/input5MobilenetEdgeTPU/expanded_conv_13/expand/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
IMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gamma
?
=MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gamma*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gamma*
T0
?
IMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/beta
VariableV2*
dtype0*
shared_name *
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/beta
?
>MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/beta*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_mean
VariableV2*
dtype0*
shared_name *
shape:?*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_mean
?
EMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_mean/Initializer/zeros*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
CMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_mean*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_mean*
T0
?
SMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_variance*
use_locking(
?
GMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_variance*
T0*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_variance
?
CMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_13/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
e
8MobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
-MobilenetEdgeTPU/expanded_conv_13/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_13/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_13/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_13/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"            *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights*
dtype0
?
_MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights*
dtype0
?
aMobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights*
dtype0
?
jMobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normal/shape*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights*
dtype0*
seed2 *
T0*

seed 
?
^MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights
?
ZMobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normal/mean*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights*
T0
?
=MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights
VariableV2*
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights*
dtype0*
shared_name 
?
DMobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/Initializer/truncated_normal*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights*
use_locking(*
T0*
validate_shape(
?
BMobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights*
T0
x
;MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise/ShapeConst*%
valueB"            *
dtype0
x
CMobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0
?
5MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_13/expansion_outputBMobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise_weights/read*
strides
*
T0*
paddingSAME*
data_formatNHWC*
	dilations

?
LMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gamma*
dtype0
?
;MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gamma
VariableV2*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gamma*
dtype0*
shared_name *
shape:?*
	container 
?
BMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gamma*
use_locking(
?
@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gamma*
T0*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gamma
?
LMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/beta*
dtype0
?
:MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/beta
VariableV2*
dtype0*
shared_name *
shape:?*
	container *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/beta
?
AMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/beta
?
?MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/beta*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/beta*
T0
?
SMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_mean*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_mean*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_mean*
use_locking(
?
FMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_mean*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_mean*
T0
?
VMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_variance*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_variance*
dtype0*
shared_name 
?
LMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_variance*
use_locking(
?
JMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_variance*
T0*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/moving_variance/read*
epsilon%o?:*
U0*
T0*
data_formatNHWC*
is_training( 
h
;MobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
0MobilenetEdgeTPU/expanded_conv_13/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_13/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_13/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_13/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"         `   *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_13/project/weights
?
SMobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_13/project/weights*
dtype0
?
UMobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_13/project/weights
?
^MobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normal/shape*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_13/project/weights*
dtype0*
seed2 *
T0*

seed 
?
RMobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_13/project/weights
?
NMobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normal/mean*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_13/project/weights
?
1MobilenetEdgeTPU/expanded_conv_13/project/weights
VariableV2*
shape:?`*
	container *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_13/project/weights*
dtype0*
shared_name 
?
8MobilenetEdgeTPU/expanded_conv_13/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_13/project/weightsNMobilenetEdgeTPU/expanded_conv_13/project/weights/Initializer/truncated_normal*
use_locking(*
T0*
validate_shape(*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_13/project/weights
?
6MobilenetEdgeTPU/expanded_conv_13/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_13/project/weights*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_13/project/weights
l
7MobilenetEdgeTPU/expanded_conv_13/project/dilation_rateConst*
valueB"      *
dtype0
?
0MobilenetEdgeTPU/expanded_conv_13/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_13/depthwise_output6MobilenetEdgeTPU/expanded_conv_13/project/weights/read*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
strides
*
T0*
data_formatNHWC
?
JMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gamma/Initializer/onesConst*
valueB`*  ??*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gamma*
dtype0
?
9MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gamma
VariableV2*
dtype0*
shared_name *
shape:`*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gamma
?
@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gamma*
use_locking(
?
>MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gamma*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gamma*
T0
?
JMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/beta/Initializer/zerosConst*
valueB`*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/beta
VariableV2*
shape:`*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/beta*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/beta*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/beta*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/beta
?
QMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB`*    *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_mean
?
?MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_mean
VariableV2*
shape:`*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
FMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_mean*
use_locking(*
T0
?
DMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_mean*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_mean*
T0
?
TMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_variance/Initializer/onesConst*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_variance*
dtype0*
valueB`*  ??
?
CMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_variance
VariableV2*
shape:`*
	container *V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
JMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*
validate_shape(*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_variance
?
HMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_variance*
T0*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_variance
?
DMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_13/project/Conv2D>MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
f
9MobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
2MobilenetEdgeTPU/expanded_conv_13/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_13/project/BatchNorm/FusedBatchNormV3*
T0
q
(MobilenetEdgeTPU/expanded_conv_13/outputIdentity2MobilenetEdgeTPU/expanded_conv_13/project/Identity*
T0
f
'MobilenetEdgeTPU/expanded_conv_14/inputIdentity(MobilenetEdgeTPU/expanded_conv_13/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"      `   ?  *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_14/expand/weights
?
RMobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_14/expand/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_14/expand/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normal/shape*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_14/expand/weights*
dtype0*
seed2 *
T0*

seed 
?
QMobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normal/stddev*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_14/expand/weights*
T0
?
MMobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_14/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_14/expand/weights
VariableV2*
shared_name *
shape:`?*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_14/expand/weights*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_14/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_14/expand/weightsMMobilenetEdgeTPU/expanded_conv_14/expand/weights/Initializer/truncated_normal*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_14/expand/weights*
use_locking(
?
5MobilenetEdgeTPU/expanded_conv_14/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_14/expand/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_14/expand/weights
k
6MobilenetEdgeTPU/expanded_conv_14/expand/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_14/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_14/input5MobilenetEdgeTPU/expanded_conv_14/expand/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
IMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gamma/Initializer/onesConst*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gamma*
dtype0*
valueB?*  ??
?
8MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gamma*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gamma*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gamma*
T0
?
IMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/beta
VariableV2*
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/beta*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/beta*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/beta*
T0
?
PMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_mean
?
>MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_mean*
dtype0*
shared_name 
?
EMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_mean
?
CMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
valueB?*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_variance
?
BMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_variance*
use_locking(*
T0
?
GMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_variance*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_variance*
T0
?
CMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_14/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/moving_variance/read*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0*
T0
e
8MobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
-MobilenetEdgeTPU/expanded_conv_14/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_14/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_14/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_14/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?     *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights*
dtype0
?
_MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights*
dtype0
?
aMobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights*
dtype0
?
jMobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights*
T0
?
ZMobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normal/mean*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights*
T0
?
=MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights
VariableV2*
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights*
dtype0*
shared_name 
?
DMobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/Initializer/truncated_normal*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights*
use_locking(*
T0
?
BMobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights*
T0
x
;MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise/ShapeConst*%
valueB"      ?     *
dtype0
x
CMobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise/dilation_rateConst*
dtype0*
valueB"      
?
5MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_14/expansion_outputBMobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise_weights/read*
strides
*
T0*
paddingSAME*
data_formatNHWC*
	dilations

?
LMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gamma*
dtype0
?
;MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gamma
VariableV2*
shape:?*
	container *N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gamma*
dtype0*
shared_name 
?
BMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gamma*
use_locking(
?
@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gamma*
T0*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gamma
?
LMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/beta*
dtype0
?
:MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/beta
VariableV2*
shape:?*
	container *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/beta*
dtype0*
shared_name 
?
AMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/beta*
use_locking(
?
?MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/beta*
T0*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/beta
?
SMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_mean*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_mean
VariableV2*
dtype0*
shared_name *
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_mean
?
HMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_mean
?
FMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_mean*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_mean
?
VMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_variance*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_variance*
dtype0*
shared_name 
?
LMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_variance/Initializer/ones*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_variance*
use_locking(*
T0*
validate_shape(
?
JMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_variance*
T0*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
h
;MobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
0MobilenetEdgeTPU/expanded_conv_14/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_14/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_14/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_14/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?  `   *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_14/project/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_14/project/weights
?
UMobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_14/project/weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normal/shape*
dtype0*
seed2 *
T0*

seed *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_14/project/weights
?
RMobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_14/project/weights
?
NMobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normal/mean*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_14/project/weights*
T0
?
1MobilenetEdgeTPU/expanded_conv_14/project/weights
VariableV2*
shape:?`*
	container *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_14/project/weights*
dtype0*
shared_name 
?
8MobilenetEdgeTPU/expanded_conv_14/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_14/project/weightsNMobilenetEdgeTPU/expanded_conv_14/project/weights/Initializer/truncated_normal*
validate_shape(*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_14/project/weights*
use_locking(*
T0
?
6MobilenetEdgeTPU/expanded_conv_14/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_14/project/weights*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_14/project/weights*
T0
l
7MobilenetEdgeTPU/expanded_conv_14/project/dilation_rateConst*
dtype0*
valueB"      
?
0MobilenetEdgeTPU/expanded_conv_14/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_14/depthwise_output6MobilenetEdgeTPU/expanded_conv_14/project/weights/read*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 
?
JMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gamma/Initializer/onesConst*
valueB`*  ??*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gamma*
dtype0
?
9MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gamma
VariableV2*
shape:`*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gamma*
dtype0*
shared_name 
?
@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gamma/Initializer/ones*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gamma*
use_locking(*
T0
?
>MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gamma*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gamma*
T0
?
JMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/beta/Initializer/zerosConst*
valueB`*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/beta
VariableV2*
shape:`*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/beta*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/beta/Initializer/zeros*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/beta*
use_locking(*
T0
?
=MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/beta*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/beta
?
QMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB`*    *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_mean
?
?MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_mean
VariableV2*
shape:`*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
FMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_mean*
use_locking(
?
DMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_mean*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_mean*
T0
?
TMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB`*  ??*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_variance*
dtype0
?
CMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_variance
VariableV2*
shape:`*
	container *V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
JMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_variance/Initializer/ones*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_variance*
use_locking(*
T0*
validate_shape(
?
HMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_variance*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_variance*
T0
?
DMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_14/project/Conv2D>MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/moving_variance/read*
epsilon%o?:*
U0*
T0*
data_formatNHWC*
is_training( 
f
9MobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
2MobilenetEdgeTPU/expanded_conv_14/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_14/project/BatchNorm/FusedBatchNormV3*
T0
?
%MobilenetEdgeTPU/expanded_conv_14/addAddV22MobilenetEdgeTPU/expanded_conv_14/project/Identity'MobilenetEdgeTPU/expanded_conv_14/input*
T0
d
(MobilenetEdgeTPU/expanded_conv_14/outputIdentity%MobilenetEdgeTPU/expanded_conv_14/add*
T0
f
'MobilenetEdgeTPU/expanded_conv_15/inputIdentity(MobilenetEdgeTPU/expanded_conv_14/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      `   ?  *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_15/expand/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_15/expand/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_15/expand/weights*
dtype0*
valueB
 *?Q?=
?
]MobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_15/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_15/expand/weights
?
MMobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_15/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_15/expand/weights
VariableV2*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_15/expand/weights*
dtype0*
shared_name *
shape:`?*
	container 
?
7MobilenetEdgeTPU/expanded_conv_15/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_15/expand/weightsMMobilenetEdgeTPU/expanded_conv_15/expand/weights/Initializer/truncated_normal*
use_locking(*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_15/expand/weights
?
5MobilenetEdgeTPU/expanded_conv_15/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_15/expand/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_15/expand/weights
k
6MobilenetEdgeTPU/expanded_conv_15/expand/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_15/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_15/input5MobilenetEdgeTPU/expanded_conv_15/expand/weights/read*
paddingSAME*
	dilations
*
strides
*
T0*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
IMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gamma
VariableV2*
shared_name *
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gamma*
dtype0
?
?MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gamma*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gamma*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gamma*
T0
?
IMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/beta/Initializer/zerosConst*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/beta*
dtype0*
valueB?*    
?
7MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/beta
VariableV2*
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/beta/Initializer/zeros*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/beta*
use_locking(*
T0
?
<MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_mean
VariableV2*
shared_name *
shape:?*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_mean*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_mean/Initializer/zeros*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
CMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_variance*
use_locking(*
T0
?
GMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_variance*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_variance*
T0
?
CMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_15/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/moving_variance/read*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0*
T0
e
8MobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
-MobilenetEdgeTPU/expanded_conv_15/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_15/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_15/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_15/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"      ?     *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights
?
_MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights*
dtype0
?
aMobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights*
dtype0
?
jMobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normal/shape*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights*
dtype0*
seed2 *
T0*

seed 
?
^MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights
?
ZMobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normal/mean*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights
?
=MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights
VariableV2*
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights*
dtype0*
shared_name 
?
DMobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/Initializer/truncated_normal*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights*
use_locking(*
T0*
validate_shape(
?
BMobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights*
T0
x
;MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise/ShapeConst*%
valueB"      ?     *
dtype0
x
CMobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0
?
5MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_15/expansion_outputBMobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise_weights/read*
strides
*
T0*
paddingSAME*
data_formatNHWC*
	dilations

?
LMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gamma*
dtype0
?
;MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gamma
VariableV2*
shape:?*
	container *N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gamma*
dtype0*
shared_name 
?
BMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gamma*
use_locking(
?
@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gamma*
T0*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gamma
?
LMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/beta/Initializer/zerosConst*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/beta*
dtype0*
valueB?*    
?
:MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/beta
VariableV2*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/beta*
dtype0*
shared_name *
shape:?*
	container 
?
AMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/beta*
use_locking(
?
?MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/beta*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/beta*
T0
?
SMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_mean*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_mean*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_mean
?
FMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_mean*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_mean
?
VMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_variance*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_variance*
dtype0*
shared_name 
?
LMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_variance*
use_locking(
?
JMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_variance*
T0*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
h
;MobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
0MobilenetEdgeTPU/expanded_conv_15/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_15/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_15/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_15/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?  `   *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_15/project/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_15/project/weights*
dtype0
?
UMobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_15/project/weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_15/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normal/stddev*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_15/project/weights*
T0
?
NMobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normal/mean*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_15/project/weights*
T0
?
1MobilenetEdgeTPU/expanded_conv_15/project/weights
VariableV2*
shape:?`*
	container *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_15/project/weights*
dtype0*
shared_name 
?
8MobilenetEdgeTPU/expanded_conv_15/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_15/project/weightsNMobilenetEdgeTPU/expanded_conv_15/project/weights/Initializer/truncated_normal*
T0*
validate_shape(*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_15/project/weights*
use_locking(
?
6MobilenetEdgeTPU/expanded_conv_15/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_15/project/weights*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_15/project/weights
l
7MobilenetEdgeTPU/expanded_conv_15/project/dilation_rateConst*
valueB"      *
dtype0
?
0MobilenetEdgeTPU/expanded_conv_15/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_15/depthwise_output6MobilenetEdgeTPU/expanded_conv_15/project/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
JMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gamma/Initializer/onesConst*
dtype0*
valueB`*  ??*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gamma
?
9MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gamma
VariableV2*
shape:`*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gamma*
dtype0*
shared_name 
?
@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gamma*
use_locking(
?
>MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gamma*
T0*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gamma
?
JMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/beta/Initializer/zerosConst*
valueB`*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/beta
VariableV2*
dtype0*
shared_name *
shape:`*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/beta
?
?MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/beta
?
=MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/beta*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/beta*
T0
?
QMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB`*    *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_mean*
dtype0
?
?MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_mean
VariableV2*
dtype0*
shared_name *
shape:`*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_mean
?
FMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_mean*
use_locking(
?
DMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_mean*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_mean*
T0
?
TMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_variance/Initializer/onesConst*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_variance*
dtype0*
valueB`*  ??
?
CMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_variance
VariableV2*
shape:`*
	container *V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
JMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_variance*
use_locking(
?
HMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_variance*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_variance*
T0
?
DMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_15/project/Conv2D>MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
f
9MobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
2MobilenetEdgeTPU/expanded_conv_15/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_15/project/BatchNorm/FusedBatchNormV3*
T0
?
%MobilenetEdgeTPU/expanded_conv_15/addAddV22MobilenetEdgeTPU/expanded_conv_15/project/Identity'MobilenetEdgeTPU/expanded_conv_15/input*
T0
d
(MobilenetEdgeTPU/expanded_conv_15/outputIdentity%MobilenetEdgeTPU/expanded_conv_15/add*
T0
f
'MobilenetEdgeTPU/expanded_conv_16/inputIdentity(MobilenetEdgeTPU/expanded_conv_15/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      `   ?  *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_16/expand/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_16/expand/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_16/expand/weights*
dtype0*
valueB
 *?Q?=
?
]MobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_16/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_16/expand/weights
?
MMobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_16/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_16/expand/weights
VariableV2*
shape:`?*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_16/expand/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_16/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_16/expand/weightsMMobilenetEdgeTPU/expanded_conv_16/expand/weights/Initializer/truncated_normal*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_16/expand/weights*
use_locking(*
T0
?
5MobilenetEdgeTPU/expanded_conv_16/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_16/expand/weights*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_16/expand/weights*
T0
k
6MobilenetEdgeTPU/expanded_conv_16/expand/dilation_rateConst*
dtype0*
valueB"      
?
/MobilenetEdgeTPU/expanded_conv_16/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_16/input5MobilenetEdgeTPU/expanded_conv_16/expand/weights/read*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 
?
IMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gamma/Initializer/onesConst*
dtype0*
valueB?*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gamma
?
8MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gamma
VariableV2*
dtype0*
shared_name *
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gamma
?
?MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gamma*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gamma*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gamma
?
IMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/beta
VariableV2*
dtype0*
shared_name *
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/beta
?
>MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/beta*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/beta*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/beta*
T0
?
PMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_mean
?
>MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_mean
VariableV2*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_mean*
dtype0*
shared_name *
shape:?*
	container 
?
EMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_mean*
use_locking(
?
CMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
valueB?*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_variance
?
BMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_variance
VariableV2*
shared_name *
shape:?*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_variance*
dtype0
?
IMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_variance/Initializer/ones*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_variance*
use_locking(*
T0*
validate_shape(
?
GMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_variance*
T0*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_variance
?
CMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_16/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
e
8MobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
-MobilenetEdgeTPU/expanded_conv_16/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_16/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_16/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_16/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"      ?     *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights
?
_MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights*
dtype0
?
aMobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights*
dtype0
?
jMobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights*
T0
?
ZMobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normal/mean*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights
?
=MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights
VariableV2*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights*
dtype0*
shared_name *
shape:?*
	container 
?
DMobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/Initializer/truncated_normal*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights*
use_locking(*
T0*
validate_shape(
?
BMobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights
x
;MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise/ShapeConst*
dtype0*%
valueB"      ?     
x
CMobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise/dilation_rateConst*
dtype0*
valueB"      
?
5MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_16/expansion_outputBMobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise_weights/read*
strides
*
T0*
paddingSAME*
data_formatNHWC*
	dilations

?
LMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gamma*
dtype0
?
;MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gamma
VariableV2*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gamma*
dtype0*
shared_name *
shape:?*
	container 
?
BMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gamma*
use_locking(
?
@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gamma*
T0*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gamma
?
LMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/beta*
dtype0
?
:MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/beta
VariableV2*
shape:?*
	container *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/beta*
dtype0*
shared_name 
?
AMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/beta/Initializer/zeros*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/beta*
use_locking(*
T0
?
?MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/beta*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/beta*
T0
?
SMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_mean
?
AMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_mean
VariableV2*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_mean*
dtype0*
shared_name *
shape:?*
	container 
?
HMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_mean*
use_locking(
?
FMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_mean*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_mean*
T0
?
VMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_variance/Initializer/onesConst*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_variance*
dtype0*
valueB?*  ??
?
EMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_variance*
dtype0*
shared_name 
?
LMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_variance*
use_locking(
?
JMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_variance*
T0*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
h
;MobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
0MobilenetEdgeTPU/expanded_conv_16/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_16/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_16/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_16/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normal/shapeConst*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_16/project/weights*
dtype0*%
valueB"      ?  `   
?
SMobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_16/project/weights*
dtype0
?
UMobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_16/project/weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normal/shape*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_16/project/weights*
dtype0*
seed2 *
T0*

seed 
?
RMobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_16/project/weights
?
NMobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normal/mean*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_16/project/weights
?
1MobilenetEdgeTPU/expanded_conv_16/project/weights
VariableV2*
shape:?`*
	container *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_16/project/weights*
dtype0*
shared_name 
?
8MobilenetEdgeTPU/expanded_conv_16/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_16/project/weightsNMobilenetEdgeTPU/expanded_conv_16/project/weights/Initializer/truncated_normal*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_16/project/weights*
use_locking(*
T0*
validate_shape(
?
6MobilenetEdgeTPU/expanded_conv_16/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_16/project/weights*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_16/project/weights
l
7MobilenetEdgeTPU/expanded_conv_16/project/dilation_rateConst*
valueB"      *
dtype0
?
0MobilenetEdgeTPU/expanded_conv_16/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_16/depthwise_output6MobilenetEdgeTPU/expanded_conv_16/project/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
JMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gamma/Initializer/onesConst*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gamma*
dtype0*
valueB`*  ??
?
9MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gamma
VariableV2*
dtype0*
shared_name *
shape:`*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gamma
?
@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gamma*
use_locking(
?
>MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gamma*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gamma*
T0
?
JMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/beta/Initializer/zerosConst*
valueB`*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/beta
VariableV2*
shape:`*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/beta*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/beta/Initializer/zeros*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/beta*
use_locking(*
T0
?
=MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/beta*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/beta
?
QMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB`*    *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_mean*
dtype0
?
?MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_mean
VariableV2*
dtype0*
shared_name *
shape:`*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_mean
?
FMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_mean*
use_locking(
?
DMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_mean*
T0*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_mean
?
TMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_variance/Initializer/onesConst*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_variance*
dtype0*
valueB`*  ??
?
CMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_variance
VariableV2*
shape:`*
	container *V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
JMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_variance*
use_locking(*
T0
?
HMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_variance*
T0*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_variance
?
DMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_16/project/Conv2D>MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/moving_variance/read*
epsilon%o?:*
U0*
T0*
data_formatNHWC*
is_training( 
f
9MobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
2MobilenetEdgeTPU/expanded_conv_16/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_16/project/BatchNorm/FusedBatchNormV3*
T0
?
%MobilenetEdgeTPU/expanded_conv_16/addAddV22MobilenetEdgeTPU/expanded_conv_16/project/Identity'MobilenetEdgeTPU/expanded_conv_16/input*
T0
d
(MobilenetEdgeTPU/expanded_conv_16/outputIdentity%MobilenetEdgeTPU/expanded_conv_16/add*
T0
f
'MobilenetEdgeTPU/expanded_conv_17/inputIdentity(MobilenetEdgeTPU/expanded_conv_16/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_17/expand/weights*
dtype0*%
valueB"      `      
?
RMobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_17/expand/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_17/expand/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normal/shape*
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_17/expand/weights*
dtype0*
seed2 
?
QMobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normal/stddev*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_17/expand/weights*
T0
?
MMobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_17/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_17/expand/weights
VariableV2*
shape:`?*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_17/expand/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_17/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_17/expand/weightsMMobilenetEdgeTPU/expanded_conv_17/expand/weights/Initializer/truncated_normal*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_17/expand/weights*
use_locking(
?
5MobilenetEdgeTPU/expanded_conv_17/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_17/expand/weights*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_17/expand/weights*
T0
k
6MobilenetEdgeTPU/expanded_conv_17/expand/dilation_rateConst*
dtype0*
valueB"      
?
/MobilenetEdgeTPU/expanded_conv_17/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_17/input5MobilenetEdgeTPU/expanded_conv_17/expand/weights/read*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
strides
*
T0
?
IMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gamma/Initializer/ones*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gamma*
use_locking(*
T0
?
=MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gamma*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gamma
?
IMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/beta/Initializer/zerosConst*
dtype0*
valueB?*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/beta
?
7MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/beta
VariableV2*
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/beta*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_mean
VariableV2*
shared_name *
shape:?*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_mean*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_mean*
use_locking(
?
CMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_mean*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_mean*
T0
?
SMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_variance*
use_locking(*
T0
?
GMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_variance*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_variance*
T0
?
CMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_17/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
e
8MobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
-MobilenetEdgeTPU/expanded_conv_17/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_17/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_17/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_17/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"            *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights*
dtype0
?
_MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights*
dtype0*
valueB
 *    
?
aMobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights*
dtype0*
valueB
 *?Q?=
?
jMobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normal/shape*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights*
dtype0*
seed2 *
T0*

seed 
?
^MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights
?
ZMobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normal/mean*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights
?
=MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights
VariableV2*
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights*
dtype0*
shared_name 
?
DMobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/Initializer/truncated_normal*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights*
use_locking(*
T0*
validate_shape(
?
BMobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights
x
;MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise/ShapeConst*%
valueB"            *
dtype0
x
CMobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise/dilation_rateConst*
dtype0*
valueB"      
?
5MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_17/expansion_outputBMobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise_weights/read*
strides
*
T0*
paddingSAME*
data_formatNHWC*
	dilations

?
LMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gamma*
dtype0
?
;MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gamma
VariableV2*
shape:?*
	container *N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gamma*
dtype0*
shared_name 
?
BMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gamma*
use_locking(
?
@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gamma*
T0*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gamma
?
LMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/beta*
dtype0
?
:MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/beta
VariableV2*
shape:?*
	container *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/beta*
dtype0*
shared_name 
?
AMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/beta/Initializer/zeros*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/beta*
use_locking(*
T0
?
?MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/beta*
T0*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/beta
?
SMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_mean*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_mean*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_mean*
use_locking(*
T0
?
FMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_mean*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_mean
?
VMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_variance*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_variance*
dtype0*
shared_name 
?
LMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_variance*
use_locking(*
T0
?
JMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_variance*
T0*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
h
;MobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
0MobilenetEdgeTPU/expanded_conv_17/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_17/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_17/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_17/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"         ?   *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_17/project/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_17/project/weights*
dtype0
?
UMobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_17/project/weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normal/shape*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_17/project/weights*
dtype0*
seed2 *
T0*

seed 
?
RMobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_17/project/weights
?
NMobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normal/mean*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_17/project/weights
?
1MobilenetEdgeTPU/expanded_conv_17/project/weights
VariableV2*
shape:??*
	container *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_17/project/weights*
dtype0*
shared_name 
?
8MobilenetEdgeTPU/expanded_conv_17/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_17/project/weightsNMobilenetEdgeTPU/expanded_conv_17/project/weights/Initializer/truncated_normal*
T0*
validate_shape(*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_17/project/weights*
use_locking(
?
6MobilenetEdgeTPU/expanded_conv_17/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_17/project/weights*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_17/project/weights*
T0
l
7MobilenetEdgeTPU/expanded_conv_17/project/dilation_rateConst*
dtype0*
valueB"      
?
0MobilenetEdgeTPU/expanded_conv_17/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_17/depthwise_output6MobilenetEdgeTPU/expanded_conv_17/project/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
JMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gamma*
dtype0
?
9MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gamma
VariableV2*
shared_name *
shape:?*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gamma*
dtype0
?
@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gamma*
use_locking(
?
>MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gamma*
T0*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gamma
?
JMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/beta
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/beta*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/beta/Initializer/zeros*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/beta*
use_locking(*
T0*
validate_shape(
?
=MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/beta*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/beta
?
QMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_mean
?
?MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
FMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_mean*
use_locking(
?
DMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_mean*
T0*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_mean
?
TMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_variance*
dtype0
?
CMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
JMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_variance*
use_locking(
?
HMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_variance*
T0*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_variance
?
DMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_17/project/Conv2D>MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
f
9MobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
2MobilenetEdgeTPU/expanded_conv_17/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_17/project/BatchNorm/FusedBatchNormV3*
T0
q
(MobilenetEdgeTPU/expanded_conv_17/outputIdentity2MobilenetEdgeTPU/expanded_conv_17/project/Identity*
T0
f
'MobilenetEdgeTPU/expanded_conv_18/inputIdentity(MobilenetEdgeTPU/expanded_conv_17/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?   ?  *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_18/expand/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_18/expand/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_18/expand/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_18/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_18/expand/weights
?
MMobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_18/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_18/expand/weights
VariableV2*
dtype0*
shared_name *
shape:??*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_18/expand/weights
?
7MobilenetEdgeTPU/expanded_conv_18/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_18/expand/weightsMMobilenetEdgeTPU/expanded_conv_18/expand/weights/Initializer/truncated_normal*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_18/expand/weights*
use_locking(
?
5MobilenetEdgeTPU/expanded_conv_18/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_18/expand/weights*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_18/expand/weights*
T0
k
6MobilenetEdgeTPU/expanded_conv_18/expand/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_18/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_18/input5MobilenetEdgeTPU/expanded_conv_18/expand/weights/read*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides

?
IMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gamma/Initializer/ones*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gamma*
use_locking(*
T0
?
=MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gamma*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gamma*
T0
?
IMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/beta
VariableV2*
dtype0*
shared_name *
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/beta
?
>MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/beta*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/beta*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/beta*
T0
?
PMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_mean/Initializer/zerosConst*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_mean*
dtype0*
valueB?*    
?
>MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_mean
VariableV2*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_mean*
dtype0*
shared_name *
shape:?*
	container 
?
EMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_mean
?
CMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_variance
?
GMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_variance*
T0*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_variance
?
CMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_18/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/moving_variance/read*
epsilon%o?:*
U0*
T0*
data_formatNHWC*
is_training( 
e
8MobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
-MobilenetEdgeTPU/expanded_conv_18/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_18/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_18/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_18/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?     *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights*
dtype0
?
_MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights
?
aMobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights*
dtype0
?
jMobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights*
T0
?
ZMobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normal/mean*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights
?
=MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights
VariableV2*
shared_name *
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights*
dtype0
?
DMobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/Initializer/truncated_normal*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights*
use_locking(*
T0
?
BMobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights
x
;MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise/ShapeConst*%
valueB"      ?     *
dtype0
x
CMobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0
?
5MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_18/expansion_outputBMobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise_weights/read*
strides
*
T0*
paddingSAME*
data_formatNHWC*
	dilations

?
LMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gamma*
dtype0
?
;MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gamma
VariableV2*
shared_name *
shape:?*
	container *N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gamma*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gamma*
use_locking(
?
@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gamma*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gamma*
T0
?
LMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/beta*
dtype0
?
:MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/beta
VariableV2*
shape:?*
	container *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/beta*
dtype0*
shared_name 
?
AMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/beta*
use_locking(
?
?MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/beta*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/beta*
T0
?
SMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_mean*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_mean*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_mean*
use_locking(
?
FMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_mean*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_mean
?
VMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_variance*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_variance
VariableV2*
shared_name *
shape:?*
	container *X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_variance*
dtype0
?
LMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_variance*
use_locking(
?
JMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_variance*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_variance*
T0
?
FMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/moving_variance/read*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0*
T0
h
;MobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
0MobilenetEdgeTPU/expanded_conv_18/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_18/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_18/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_18/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?  ?   *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_18/project/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_18/project/weights*
dtype0
?
UMobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_18/project/weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_18/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_18/project/weights
?
NMobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normal/mean*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_18/project/weights
?
1MobilenetEdgeTPU/expanded_conv_18/project/weights
VariableV2*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_18/project/weights*
dtype0*
shared_name *
shape:??*
	container 
?
8MobilenetEdgeTPU/expanded_conv_18/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_18/project/weightsNMobilenetEdgeTPU/expanded_conv_18/project/weights/Initializer/truncated_normal*
T0*
validate_shape(*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_18/project/weights*
use_locking(
?
6MobilenetEdgeTPU/expanded_conv_18/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_18/project/weights*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_18/project/weights
l
7MobilenetEdgeTPU/expanded_conv_18/project/dilation_rateConst*
valueB"      *
dtype0
?
0MobilenetEdgeTPU/expanded_conv_18/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_18/depthwise_output6MobilenetEdgeTPU/expanded_conv_18/project/weights/read*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 
?
JMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gamma*
dtype0
?
9MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gamma
VariableV2*
shape:?*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gamma*
dtype0*
shared_name 
?
@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gamma*
use_locking(
?
>MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gamma*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gamma*
T0
?
JMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/beta
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/beta*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/beta
?
=MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/beta*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/beta*
T0
?
QMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_mean/Initializer/zerosConst*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_mean*
dtype0*
valueB?*    
?
?MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
FMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_mean*
use_locking(
?
DMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_mean*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_mean*
T0
?
TMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_variance*
dtype0
?
CMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_variance
VariableV2*
dtype0*
shared_name *
shape:?*
	container *V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_variance
?
JMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_variance*
use_locking(
?
HMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_variance*
T0*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_variance
?
DMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_18/project/Conv2D>MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
f
9MobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
2MobilenetEdgeTPU/expanded_conv_18/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_18/project/BatchNorm/FusedBatchNormV3*
T0
?
%MobilenetEdgeTPU/expanded_conv_18/addAddV22MobilenetEdgeTPU/expanded_conv_18/project/Identity'MobilenetEdgeTPU/expanded_conv_18/input*
T0
d
(MobilenetEdgeTPU/expanded_conv_18/outputIdentity%MobilenetEdgeTPU/expanded_conv_18/add*
T0
f
'MobilenetEdgeTPU/expanded_conv_19/inputIdentity(MobilenetEdgeTPU/expanded_conv_18/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?   ?  *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_19/expand/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_19/expand/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_19/expand/weights
?
]MobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_19/expand/weights*
dtype0
?
QMobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_19/expand/weights
?
MMobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_19/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_19/expand/weights
VariableV2*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_19/expand/weights*
dtype0*
shared_name *
shape:??*
	container 
?
7MobilenetEdgeTPU/expanded_conv_19/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_19/expand/weightsMMobilenetEdgeTPU/expanded_conv_19/expand/weights/Initializer/truncated_normal*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_19/expand/weights*
use_locking(*
T0
?
5MobilenetEdgeTPU/expanded_conv_19/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_19/expand/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_19/expand/weights
k
6MobilenetEdgeTPU/expanded_conv_19/expand/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_19/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_19/input5MobilenetEdgeTPU/expanded_conv_19/expand/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
IMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gamma
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gamma/Initializer/ones*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gamma*
use_locking(*
T0*
validate_shape(
?
=MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gamma*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gamma
?
IMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/beta
VariableV2*
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/beta*
dtype0*
shared_name 
?
>MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/beta/Initializer/zeros*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/beta*
use_locking(*
T0
?
<MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_mean/Initializer/zerosConst*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_mean*
dtype0*
valueB?*    
?
>MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_mean*
dtype0*
shared_name 
?
EMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_mean*
use_locking(*
T0
?
CMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_mean
?
SMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_variance*
use_locking(
?
GMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_variance*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_variance*
T0
?
CMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_19/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
e
8MobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
-MobilenetEdgeTPU/expanded_conv_19/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_19/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_19/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_19/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?     *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights*
dtype0
?
_MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights*
dtype0*
valueB
 *    
?
aMobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights*
dtype0
?
jMobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normal/shape*
dtype0*
seed2 *
T0*

seed *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights
?
^MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights*
T0
?
ZMobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normal/mean*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights*
T0
?
=MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights
VariableV2*
dtype0*
shared_name *
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights
?
DMobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/Initializer/truncated_normal*
use_locking(*
T0*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights
?
BMobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights
x
;MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise/ShapeConst*%
valueB"      ?     *
dtype0
x
CMobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0
?
5MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_19/expansion_outputBMobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise_weights/read*
	dilations
*
strides
*
T0*
paddingSAME*
data_formatNHWC
?
LMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gamma/Initializer/onesConst*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gamma*
dtype0*
valueB?*  ??
?
;MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gamma
VariableV2*
shape:?*
	container *N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gamma*
dtype0*
shared_name 
?
BMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gamma*
use_locking(
?
@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gamma*
T0*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gamma
?
LMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/beta/Initializer/zerosConst*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/beta*
dtype0*
valueB?*    
?
:MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/beta
VariableV2*
shape:?*
	container *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/beta*
dtype0*
shared_name 
?
AMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/beta
?
?MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/beta*
T0*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/beta
?
SMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_mean
?
AMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_mean
VariableV2*
shared_name *
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_mean*
dtype0
?
HMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_mean
?
FMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_mean*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_mean*
T0
?
VMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_variance*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_variance*
dtype0*
shared_name 
?
LMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_variance*
use_locking(
?
JMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_variance*
T0*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
h
;MobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
0MobilenetEdgeTPU/expanded_conv_19/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_19/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_19/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_19/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normal/shapeConst*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_19/project/weights*
dtype0*%
valueB"      ?  ?   
?
SMobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_19/project/weights*
dtype0
?
UMobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_19/project/weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normal/shape*
T0*

seed *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_19/project/weights*
dtype0*
seed2 
?
RMobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normal/stddev*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_19/project/weights*
T0
?
NMobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normal/mean*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_19/project/weights*
T0
?
1MobilenetEdgeTPU/expanded_conv_19/project/weights
VariableV2*
shared_name *
shape:??*
	container *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_19/project/weights*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_19/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_19/project/weightsNMobilenetEdgeTPU/expanded_conv_19/project/weights/Initializer/truncated_normal*
T0*
validate_shape(*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_19/project/weights*
use_locking(
?
6MobilenetEdgeTPU/expanded_conv_19/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_19/project/weights*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_19/project/weights*
T0
l
7MobilenetEdgeTPU/expanded_conv_19/project/dilation_rateConst*
valueB"      *
dtype0
?
0MobilenetEdgeTPU/expanded_conv_19/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_19/depthwise_output6MobilenetEdgeTPU/expanded_conv_19/project/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
JMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gamma/Initializer/onesConst*
dtype0*
valueB?*  ??*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gamma
?
9MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gamma
VariableV2*
shape:?*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gamma*
dtype0*
shared_name 
?
@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gamma/Initializer/ones*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gamma*
use_locking(*
T0
?
>MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gamma*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gamma*
T0
?
JMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/beta
VariableV2*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/beta*
dtype0*
shared_name *
shape:?*
	container 
?
?MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/beta/Initializer/zeros*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/beta*
use_locking(*
T0*
validate_shape(
?
=MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/beta*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/beta
?
QMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_mean/Initializer/zerosConst*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_mean*
dtype0*
valueB?*    
?
?MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
FMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_mean*
use_locking(*
T0
?
DMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_mean*
T0*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_mean
?
TMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_variance*
dtype0
?
CMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
JMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_variance*
use_locking(
?
HMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_variance*
T0*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_variance
?
DMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_19/project/Conv2D>MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/moving_variance/read*
epsilon%o?:*
U0*
T0*
data_formatNHWC*
is_training( 
f
9MobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/ConstConst*
dtype0*
valueB
 *d;?
?
2MobilenetEdgeTPU/expanded_conv_19/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_19/project/BatchNorm/FusedBatchNormV3*
T0
?
%MobilenetEdgeTPU/expanded_conv_19/addAddV22MobilenetEdgeTPU/expanded_conv_19/project/Identity'MobilenetEdgeTPU/expanded_conv_19/input*
T0
d
(MobilenetEdgeTPU/expanded_conv_19/outputIdentity%MobilenetEdgeTPU/expanded_conv_19/add*
T0
f
'MobilenetEdgeTPU/expanded_conv_20/inputIdentity(MobilenetEdgeTPU/expanded_conv_19/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?   ?  *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_20/expand/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_20/expand/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_20/expand/weights
?
]MobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normal/shape*
T0*

seed *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_20/expand/weights*
dtype0*
seed2 
?
QMobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normal/stddev*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_20/expand/weights*
T0
?
MMobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_20/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_20/expand/weights
VariableV2*
shared_name *
shape:??*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_20/expand/weights*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_20/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_20/expand/weightsMMobilenetEdgeTPU/expanded_conv_20/expand/weights/Initializer/truncated_normal*
T0*
validate_shape(*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_20/expand/weights*
use_locking(
?
5MobilenetEdgeTPU/expanded_conv_20/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_20/expand/weights*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_20/expand/weights
k
6MobilenetEdgeTPU/expanded_conv_20/expand/dilation_rateConst*
valueB"      *
dtype0
?
/MobilenetEdgeTPU/expanded_conv_20/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_20/input5MobilenetEdgeTPU/expanded_conv_20/expand/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
IMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gamma/Initializer/onesConst*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gamma*
dtype0*
valueB?*  ??
?
8MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gamma
VariableV2*
shared_name *
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gamma*
dtype0
?
?MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gamma*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gamma*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gamma
?
IMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/beta*
dtype0
?
7MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/beta
VariableV2*
shared_name *
shape:?*
	container *J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/beta*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/beta*
use_locking(
?
<MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/beta*
T0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/beta
?
PMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_mean*
dtype0
?
>MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_mean*
dtype0*
shared_name 
?
EMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_mean*
use_locking(*
T0
?
CMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_mean*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_mean*
T0
?
SMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_variance*
use_locking(
?
GMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_variance*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_variance*
T0
?
CMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_20/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
e
8MobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
-MobilenetEdgeTPU/expanded_conv_20/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_20/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_20/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_20/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*
dtype0*%
valueB"      ?     *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights
?
_MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights*
dtype0*
valueB
 *    
?
aMobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights*
dtype0
?
jMobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normal/shape*
T0*

seed *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights*
dtype0*
seed2 
?
^MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights
?
ZMobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normal/mean*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights*
T0
?
=MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights
VariableV2*
shape:?*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights*
dtype0*
shared_name 
?
DMobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/Initializer/truncated_normal*
T0*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights*
use_locking(
?
BMobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights
x
;MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise/ShapeConst*%
valueB"      ?     *
dtype0
x
CMobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0
?
5MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_20/expansion_outputBMobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise_weights/read*
strides
*
T0*
paddingSAME*
data_formatNHWC*
	dilations

?
LMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gamma/Initializer/onesConst*
dtype0*
valueB?*  ??*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gamma
?
;MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gamma
VariableV2*
shape:?*
	container *N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gamma*
dtype0*
shared_name 
?
BMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gamma/Initializer/ones*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gamma*
use_locking(*
T0*
validate_shape(
?
@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gamma*
T0*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gamma
?
LMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/beta*
dtype0
?
:MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/beta
VariableV2*
shared_name *
shape:?*
	container *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/beta*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/beta
?
?MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/beta*
T0*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/beta
?
SMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_mean*
dtype0
?
AMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_mean*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_mean*
use_locking(
?
FMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_mean*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_mean*
T0
?
VMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_variance*
dtype0
?
EMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_variance
VariableV2*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_variance*
dtype0*
shared_name *
shape:?*
	container 
?
LMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_variance*
use_locking(
?
JMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_variance*
T0*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
h
;MobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
0MobilenetEdgeTPU/expanded_conv_20/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_20/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_20/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_20/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?  ?   *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_20/project/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_20/project/weights*
dtype0
?
UMobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_20/project/weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_20/project/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_20/project/weights
?
NMobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normal/mean*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_20/project/weights*
T0
?
1MobilenetEdgeTPU/expanded_conv_20/project/weights
VariableV2*
shape:??*
	container *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_20/project/weights*
dtype0*
shared_name 
?
8MobilenetEdgeTPU/expanded_conv_20/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_20/project/weightsNMobilenetEdgeTPU/expanded_conv_20/project/weights/Initializer/truncated_normal*
use_locking(*
T0*
validate_shape(*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_20/project/weights
?
6MobilenetEdgeTPU/expanded_conv_20/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_20/project/weights*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_20/project/weights
l
7MobilenetEdgeTPU/expanded_conv_20/project/dilation_rateConst*
valueB"      *
dtype0
?
0MobilenetEdgeTPU/expanded_conv_20/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_20/depthwise_output6MobilenetEdgeTPU/expanded_conv_20/project/weights/read*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides

?
JMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gamma*
dtype0
?
9MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gamma
VariableV2*
dtype0*
shared_name *
shape:?*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gamma
?
@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gamma/Initializer/ones*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gamma*
use_locking(*
T0
?
>MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gamma*
T0*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gamma
?
JMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/beta
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/beta*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/beta/Initializer/zeros*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/beta*
use_locking(*
T0
?
=MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/beta*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/beta
?
QMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_mean/Initializer/zerosConst*
valueB?*    *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_mean*
dtype0
?
?MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
FMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_mean*
use_locking(
?
DMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_mean*
T0*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_mean
?
TMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_variance*
dtype0
?
CMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_variance
VariableV2*
shape:?*
	container *V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_variance*
dtype0*
shared_name 
?
JMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_variance*
use_locking(*
T0
?
HMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_variance*
T0*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_variance
?
DMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_20/project/Conv2D>MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
f
9MobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
2MobilenetEdgeTPU/expanded_conv_20/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_20/project/BatchNorm/FusedBatchNormV3*
T0
?
%MobilenetEdgeTPU/expanded_conv_20/addAddV22MobilenetEdgeTPU/expanded_conv_20/project/Identity'MobilenetEdgeTPU/expanded_conv_20/input*
T0
d
(MobilenetEdgeTPU/expanded_conv_20/outputIdentity%MobilenetEdgeTPU/expanded_conv_20/add*
T0
f
'MobilenetEdgeTPU/expanded_conv_21/inputIdentity(MobilenetEdgeTPU/expanded_conv_20/output*
T0
?
SMobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?      *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_21/expand/weights*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_21/expand/weights*
dtype0
?
TMobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_21/expand/weights*
dtype0
?
]MobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSMobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normal/shape*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_21/expand/weights*
dtype0*
seed2 *
T0*

seed 
?
QMobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normal/mulMul]MobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normal/TruncatedNormalTMobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_21/expand/weights
?
MMobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normalAddQMobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normal/mulRMobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_21/expand/weights
?
0MobilenetEdgeTPU/expanded_conv_21/expand/weights
VariableV2*
shape:??
*
	container *C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_21/expand/weights*
dtype0*
shared_name 
?
7MobilenetEdgeTPU/expanded_conv_21/expand/weights/AssignAssign0MobilenetEdgeTPU/expanded_conv_21/expand/weightsMMobilenetEdgeTPU/expanded_conv_21/expand/weights/Initializer/truncated_normal*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_21/expand/weights*
use_locking(*
T0*
validate_shape(
?
5MobilenetEdgeTPU/expanded_conv_21/expand/weights/readIdentity0MobilenetEdgeTPU/expanded_conv_21/expand/weights*C
_class9
75loc:@MobilenetEdgeTPU/expanded_conv_21/expand/weights*
T0
k
6MobilenetEdgeTPU/expanded_conv_21/expand/dilation_rateConst*
dtype0*
valueB"      
?
/MobilenetEdgeTPU/expanded_conv_21/expand/Conv2DConv2D'MobilenetEdgeTPU/expanded_conv_21/input5MobilenetEdgeTPU/expanded_conv_21/expand/weights/read*
use_cudnn_on_gpu(*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 
?
YMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*
valueB:?
*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma*
dtype0
?
OMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma/Initializer/ones/ConstConst*
valueB
 *  ??*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma*
dtype0
?
IMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma/Initializer/onesFillYMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma/Initializer/ones/shape_as_tensorOMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma/Initializer/ones/Const*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma*
T0*

index_type0
?
8MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma
VariableV2*
shape:?
*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma/AssignAssign8MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gammaIMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma
?
=MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma/readIdentity8MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma
?
YMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
valueB:?
*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta*
dtype0
?
OMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta/Initializer/zeros/ConstConst*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta*
dtype0*
valueB
 *    
?
IMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta/Initializer/zerosFillYMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta/Initializer/zeros/shape_as_tensorOMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta/Initializer/zeros/Const*
T0*

index_type0*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta
?
7MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta
VariableV2*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta*
dtype0*
shared_name *
shape:?
*
	container 
?
>MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta/AssignAssign7MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/betaIMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta/Initializer/zeros*
validate_shape(*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta*
use_locking(*
T0
?
<MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta/readIdentity7MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta*J
_class@
><loc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta*
T0
?
`MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:?
*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean*
dtype0
?
VMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean*
dtype0
?
PMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean/Initializer/zerosFill`MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorVMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*

index_type0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean
?
>MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean
VariableV2*
shape:?
*
	container *Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean*
dtype0*
shared_name 
?
EMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean/AssignAssign>MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_meanPMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean/Initializer/zeros*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean*
use_locking(*
T0*
validate_shape(
?
CMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean/readIdentity>MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean*
T0*Q
_classG
ECloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean
?
cMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
valueB:?
*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance*
dtype0
?
YMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ??*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance/Initializer/onesFillcMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorYMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance/Initializer/ones/Const*
T0*

index_type0*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance
?
BMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance
VariableV2*
shape:?
*
	container *U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance*
dtype0*
shared_name 
?
IMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance/AssignAssignBMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_varianceSMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance*
use_locking(*
T0
?
GMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance/readIdentityBMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance*U
_classK
IGloc:@MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance*
T0
?
CMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/FusedBatchNormV3FusedBatchNormV3/MobilenetEdgeTPU/expanded_conv_21/expand/Conv2D=MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/gamma/read<MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/beta/readCMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_mean/readGMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
e
8MobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
-MobilenetEdgeTPU/expanded_conv_21/expand/ReluReluCMobilenetEdgeTPU/expanded_conv_21/expand/BatchNorm/FusedBatchNormV3*
T0
v
2MobilenetEdgeTPU/expanded_conv_21/expansion_outputIdentity-MobilenetEdgeTPU/expanded_conv_21/expand/Relu*
T0
?
`MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normal/shapeConst*%
valueB"            *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights*
dtype0
?
_MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights
?
aMobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normal/stddevConst*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights*
dtype0*
valueB
 *?Q?=
?
jMobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal`MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normal/shape*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights*
dtype0*
seed2 *
T0*

seed 
?
^MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normal/mulMuljMobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normal/TruncatedNormalaMobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normal/stddev*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights*
T0
?
ZMobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normalAdd^MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normal/mul_MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normal/mean*
T0*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights
?
=MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights
VariableV2*
shape:?
*
	container *P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights*
dtype0*
shared_name 
?
DMobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/AssignAssign=MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weightsZMobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/Initializer/truncated_normal*
validate_shape(*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights*
use_locking(*
T0
?
BMobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/readIdentity=MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights*P
_classF
DBloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights*
T0
x
;MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise/ShapeConst*
dtype0*%
valueB"            
x
CMobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise/dilation_rateConst*
valueB"      *
dtype0
?
5MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwiseDepthwiseConv2dNative2MobilenetEdgeTPU/expanded_conv_21/expansion_outputBMobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise_weights/read*
strides
*
T0*
paddingSAME*
data_formatNHWC*
	dilations

?
\MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
valueB:?
*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma
?
RMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma/Initializer/ones/ConstConst*
valueB
 *  ??*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma*
dtype0
?
LMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma/Initializer/onesFill\MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma/Initializer/ones/shape_as_tensorRMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma/Initializer/ones/Const*
T0*

index_type0*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma
?
;MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma
VariableV2*
shape:?
*
	container *N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma*
dtype0*
shared_name 
?
BMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma/AssignAssign;MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gammaLMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*
validate_shape(*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma
?
@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma/readIdentity;MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma*N
_classD
B@loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma*
T0
?
\MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
valueB:?
*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta*
dtype0
?
RMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta/Initializer/zeros/ConstConst*
valueB
 *    *M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta*
dtype0
?
LMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta/Initializer/zerosFill\MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta/Initializer/zeros/shape_as_tensorRMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta/Initializer/zeros/Const*
T0*

index_type0*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta
?
:MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta
VariableV2*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta*
dtype0*
shared_name *
shape:?
*
	container 
?
AMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta/AssignAssign:MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/betaLMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta/Initializer/zeros*
validate_shape(*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta*
use_locking(*
T0
?
?MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta/readIdentity:MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta*
T0*M
_classC
A?loc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta
?
cMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
valueB:?
*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean*
dtype0
?
YMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean/Initializer/zerosFillcMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorYMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean/Initializer/zeros/Const*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean*
T0*

index_type0
?
AMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean
VariableV2*
shape:?
*
	container *T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean*
dtype0*
shared_name 
?
HMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean/AssignAssignAMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_meanSMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean*
use_locking(
?
FMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean/readIdentityAMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean*
T0*T
_classJ
HFloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean
?
fMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
valueB:?
*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance
?
\MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ??*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance*
dtype0
?
VMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance/Initializer/onesFillfMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor\MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance/Initializer/ones/Const*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance*
T0*

index_type0
?
EMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance
VariableV2*
shape:?
*
	container *X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance*
dtype0*
shared_name 
?
LMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance/AssignAssignEMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_varianceVMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance*
use_locking(
?
JMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance/readIdentityEMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance*
T0*X
_classN
LJloc:@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance
?
FMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/FusedBatchNormV3FusedBatchNormV35MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise@MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/gamma/read?MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/beta/readFMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_mean/readJMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
h
;MobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
0MobilenetEdgeTPU/expanded_conv_21/depthwise/ReluReluFMobilenetEdgeTPU/expanded_conv_21/depthwise/BatchNorm/FusedBatchNormV3*
T0
y
2MobilenetEdgeTPU/expanded_conv_21/depthwise_outputIdentity0MobilenetEdgeTPU/expanded_conv_21/depthwise/Relu*
T0
?
TMobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normal/shapeConst*%
valueB"         ?   *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_21/project/weights*
dtype0
?
SMobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_21/project/weights
?
UMobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_21/project/weights*
dtype0
?
^MobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTMobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normal/shape*
dtype0*
seed2 *
T0*

seed *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_21/project/weights
?
RMobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normal/mulMul^MobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normal/TruncatedNormalUMobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normal/stddev*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_21/project/weights*
T0
?
NMobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normalAddRMobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normal/mulSMobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normal/mean*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_21/project/weights*
T0
?
1MobilenetEdgeTPU/expanded_conv_21/project/weights
VariableV2*
shared_name *
shape:?
?*
	container *D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_21/project/weights*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_21/project/weights/AssignAssign1MobilenetEdgeTPU/expanded_conv_21/project/weightsNMobilenetEdgeTPU/expanded_conv_21/project/weights/Initializer/truncated_normal*
use_locking(*
T0*
validate_shape(*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_21/project/weights
?
6MobilenetEdgeTPU/expanded_conv_21/project/weights/readIdentity1MobilenetEdgeTPU/expanded_conv_21/project/weights*
T0*D
_class:
86loc:@MobilenetEdgeTPU/expanded_conv_21/project/weights
l
7MobilenetEdgeTPU/expanded_conv_21/project/dilation_rateConst*
valueB"      *
dtype0
?
0MobilenetEdgeTPU/expanded_conv_21/project/Conv2DConv2D2MobilenetEdgeTPU/expanded_conv_21/depthwise_output6MobilenetEdgeTPU/expanded_conv_21/project/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
JMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gamma/Initializer/onesConst*
valueB?*  ??*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gamma*
dtype0
?
9MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gamma
VariableV2*
shared_name *
shape:?*
	container *L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gamma*
dtype0
?
@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gamma/AssignAssign9MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gammaJMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gamma/Initializer/ones*
T0*
validate_shape(*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gamma*
use_locking(
?
>MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gamma/readIdentity9MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gamma*
T0*L
_classB
@>loc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gamma
?
JMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/beta/Initializer/zerosConst*
valueB?*    *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/beta*
dtype0
?
8MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/beta
VariableV2*
shape:?*
	container *K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/beta*
dtype0*
shared_name 
?
?MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/beta/AssignAssign8MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/betaJMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/beta*
use_locking(
?
=MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/beta/readIdentity8MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/beta*
T0*K
_classA
?=loc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/beta
?
QMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_mean
?
?MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_mean
VariableV2*
shape:?*
	container *R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_mean*
dtype0*
shared_name 
?
FMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_mean/AssignAssign?MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_meanQMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_mean*
use_locking(
?
DMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_mean/readIdentity?MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_mean*
T0*R
_classH
FDloc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_mean
?
TMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_variance/Initializer/onesConst*
valueB?*  ??*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_variance*
dtype0
?
CMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_variance
VariableV2*
dtype0*
shared_name *
shape:?*
	container *V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_variance
?
JMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_variance/AssignAssignCMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_varianceTMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_variance/Initializer/ones*
T0*
validate_shape(*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_variance*
use_locking(
?
HMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_variance/readIdentityCMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_variance*V
_classL
JHloc:@MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_variance*
T0
?
DMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/FusedBatchNormV3FusedBatchNormV30MobilenetEdgeTPU/expanded_conv_21/project/Conv2D>MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/gamma/read=MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/beta/readDMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_mean/readHMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/moving_variance/read*
is_training( *
epsilon%o?:*
U0*
T0*
data_formatNHWC
f
9MobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
?
2MobilenetEdgeTPU/expanded_conv_21/project/IdentityIdentityDMobilenetEdgeTPU/expanded_conv_21/project/BatchNorm/FusedBatchNormV3*
T0
q
(MobilenetEdgeTPU/expanded_conv_21/outputIdentity2MobilenetEdgeTPU/expanded_conv_21/project/Identity*
T0
?
BMobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normal/shapeConst*%
valueB"      ?      *2
_class(
&$loc:@MobilenetEdgeTPU/Conv_1/weights*
dtype0
?
AMobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normal/meanConst*2
_class(
&$loc:@MobilenetEdgeTPU/Conv_1/weights*
dtype0*
valueB
 *    
?
CMobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normal/stddevConst*2
_class(
&$loc:@MobilenetEdgeTPU/Conv_1/weights*
dtype0*
valueB
 *?Q?=
?
LMobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalBMobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normal/shape*
seed2 *
T0*

seed *2
_class(
&$loc:@MobilenetEdgeTPU/Conv_1/weights*
dtype0
?
@MobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normal/mulMulLMobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normal/TruncatedNormalCMobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normal/stddev*2
_class(
&$loc:@MobilenetEdgeTPU/Conv_1/weights*
T0
?
<MobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normalAdd@MobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normal/mulAMobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normal/mean*2
_class(
&$loc:@MobilenetEdgeTPU/Conv_1/weights*
T0
?
MobilenetEdgeTPU/Conv_1/weights
VariableV2*
shape:??
*
	container *2
_class(
&$loc:@MobilenetEdgeTPU/Conv_1/weights*
dtype0*
shared_name 
?
&MobilenetEdgeTPU/Conv_1/weights/AssignAssignMobilenetEdgeTPU/Conv_1/weights<MobilenetEdgeTPU/Conv_1/weights/Initializer/truncated_normal*2
_class(
&$loc:@MobilenetEdgeTPU/Conv_1/weights*
use_locking(*
T0*
validate_shape(
?
$MobilenetEdgeTPU/Conv_1/weights/readIdentityMobilenetEdgeTPU/Conv_1/weights*
T0*2
_class(
&$loc:@MobilenetEdgeTPU/Conv_1/weights
Z
%MobilenetEdgeTPU/Conv_1/dilation_rateConst*
valueB"      *
dtype0
?
MobilenetEdgeTPU/Conv_1/Conv2DConv2D(MobilenetEdgeTPU/expanded_conv_21/output$MobilenetEdgeTPU/Conv_1/weights/read*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME*
	dilations

?
HMobilenetEdgeTPU/Conv_1/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*
valueB:?
*:
_class0
.,loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/gamma*
dtype0
?
>MobilenetEdgeTPU/Conv_1/BatchNorm/gamma/Initializer/ones/ConstConst*
valueB
 *  ??*:
_class0
.,loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/gamma*
dtype0
?
8MobilenetEdgeTPU/Conv_1/BatchNorm/gamma/Initializer/onesFillHMobilenetEdgeTPU/Conv_1/BatchNorm/gamma/Initializer/ones/shape_as_tensor>MobilenetEdgeTPU/Conv_1/BatchNorm/gamma/Initializer/ones/Const*
T0*

index_type0*:
_class0
.,loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/gamma
?
'MobilenetEdgeTPU/Conv_1/BatchNorm/gamma
VariableV2*
shape:?
*
	container *:
_class0
.,loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/gamma*
dtype0*
shared_name 
?
.MobilenetEdgeTPU/Conv_1/BatchNorm/gamma/AssignAssign'MobilenetEdgeTPU/Conv_1/BatchNorm/gamma8MobilenetEdgeTPU/Conv_1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*
validate_shape(*:
_class0
.,loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/gamma
?
,MobilenetEdgeTPU/Conv_1/BatchNorm/gamma/readIdentity'MobilenetEdgeTPU/Conv_1/BatchNorm/gamma*
T0*:
_class0
.,loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/gamma
?
HMobilenetEdgeTPU/Conv_1/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
valueB:?
*9
_class/
-+loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/beta*
dtype0
?
>MobilenetEdgeTPU/Conv_1/BatchNorm/beta/Initializer/zeros/ConstConst*
dtype0*
valueB
 *    *9
_class/
-+loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/beta
?
8MobilenetEdgeTPU/Conv_1/BatchNorm/beta/Initializer/zerosFillHMobilenetEdgeTPU/Conv_1/BatchNorm/beta/Initializer/zeros/shape_as_tensor>MobilenetEdgeTPU/Conv_1/BatchNorm/beta/Initializer/zeros/Const*

index_type0*9
_class/
-+loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/beta*
T0
?
&MobilenetEdgeTPU/Conv_1/BatchNorm/beta
VariableV2*
shape:?
*
	container *9
_class/
-+loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/beta*
dtype0*
shared_name 
?
-MobilenetEdgeTPU/Conv_1/BatchNorm/beta/AssignAssign&MobilenetEdgeTPU/Conv_1/BatchNorm/beta8MobilenetEdgeTPU/Conv_1/BatchNorm/beta/Initializer/zeros*
T0*
validate_shape(*9
_class/
-+loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/beta*
use_locking(
?
+MobilenetEdgeTPU/Conv_1/BatchNorm/beta/readIdentity&MobilenetEdgeTPU/Conv_1/BatchNorm/beta*
T0*9
_class/
-+loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/beta
?
OMobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*@
_class6
42loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean*
dtype0*
valueB:?

?
EMobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
valueB
 *    *@
_class6
42loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean*
dtype0
?
?MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean/Initializer/zerosFillOMobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorEMobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*

index_type0*@
_class6
42loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean
?
-MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean
VariableV2*
shape:?
*
	container *@
_class6
42loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean*
dtype0*
shared_name 
?
4MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean/AssignAssign-MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean?MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean/Initializer/zeros*
T0*
validate_shape(*@
_class6
42loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean*
use_locking(
?
2MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean/readIdentity-MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean*
T0*@
_class6
42loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean
?
RMobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*D
_class:
86loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance*
dtype0*
valueB:?

?
HMobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance/Initializer/ones/ConstConst*
valueB
 *  ??*D
_class:
86loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance*
dtype0
?
BMobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance/Initializer/onesFillRMobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorHMobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance/Initializer/ones/Const*
T0*

index_type0*D
_class:
86loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance
?
1MobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance
VariableV2*
shape:?
*
	container *D
_class:
86loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance*
dtype0*
shared_name 
?
8MobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance/AssignAssign1MobilenetEdgeTPU/Conv_1/BatchNorm/moving_varianceBMobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance/Initializer/ones*D
_class:
86loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance*
use_locking(*
T0*
validate_shape(
?
6MobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance/readIdentity1MobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance*
T0*D
_class:
86loc:@MobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance
?
2MobilenetEdgeTPU/Conv_1/BatchNorm/FusedBatchNormV3FusedBatchNormV3MobilenetEdgeTPU/Conv_1/Conv2D,MobilenetEdgeTPU/Conv_1/BatchNorm/gamma/read+MobilenetEdgeTPU/Conv_1/BatchNorm/beta/read2MobilenetEdgeTPU/Conv_1/BatchNorm/moving_mean/read6MobilenetEdgeTPU/Conv_1/BatchNorm/moving_variance/read*
T0*
data_formatNHWC*
is_training( *
epsilon%o?:*
U0
T
'MobilenetEdgeTPU/Conv_1/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0
a
MobilenetEdgeTPU/Conv_1/ReluRelu2MobilenetEdgeTPU/Conv_1/BatchNorm/FusedBatchNormV3*
T0
M
MobilenetEdgeTPU/embeddingIdentityMobilenetEdgeTPU/Conv_1/Relu*
T0
?
!MobilenetEdgeTPU/Logits/AvgPool2DAvgPoolMobilenetEdgeTPU/embedding*
strides
*
T0*
data_formatNHWC*
paddingVALID*
ksize

`
(MobilenetEdgeTPU/Logits/Dropout/IdentityIdentity!MobilenetEdgeTPU/Logits/AvgPool2D*
T0
?
PMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normal/shapeConst*%
valueB"         ?  *@
_class6
42loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights*
dtype0
?
OMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normal/meanConst*
dtype0*
valueB
 *    *@
_class6
42loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights
?
QMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normal/stddevConst*
valueB
 *?Q?=*@
_class6
42loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights*
dtype0
?
ZMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalPMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normal/shape*@
_class6
42loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights*
dtype0*
seed2 *
T0*

seed 
?
NMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normal/mulMulZMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normal/TruncatedNormalQMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normal/stddev*
T0*@
_class6
42loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights
?
JMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normalAddNMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normal/mulOMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normal/mean*
T0*@
_class6
42loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights
?
-MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights
VariableV2*
shape:?
?*
	container *@
_class6
42loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights*
dtype0*
shared_name 
?
4MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/AssignAssign-MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weightsJMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/Initializer/truncated_normal*
validate_shape(*@
_class6
42loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights*
use_locking(*
T0
?
2MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/readIdentity-MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights*@
_class6
42loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights*
T0
?
NMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB:?*?
_class5
31loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases
?
DMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases/Initializer/zeros/ConstConst*
valueB
 *    *?
_class5
31loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases*
dtype0
?
>MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases/Initializer/zerosFillNMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases/Initializer/zeros/shape_as_tensorDMobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases/Initializer/zeros/Const*
T0*

index_type0*?
_class5
31loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases
?
,MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases
VariableV2*
shared_name *
shape:?*
	container *?
_class5
31loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases*
dtype0
?
3MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases/AssignAssign,MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases>MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases/Initializer/zeros*
T0*
validate_shape(*?
_class5
31loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases*
use_locking(
?
1MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases/readIdentity,MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases*
T0*?
_class5
31loc:@MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases
h
3MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/dilation_rateConst*
valueB"      *
dtype0
?
,MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/Conv2DConv2D(MobilenetEdgeTPU/Logits/Dropout/Identity2MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/weights/read*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(
?
-MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/BiasAddBiasAdd,MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/Conv2D1MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/biases/read*
T0*
data_formatNHWC
z
MobilenetEdgeTPU/Logits/SqueezeSqueeze-MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/BiasAdd*
squeeze_dims
*
T0
T
MobilenetEdgeTPU/Logits/outputIdentityMobilenetEdgeTPU/Logits/Squeeze*
T0
_
*MobilenetEdgeTPU/Predictions/Reshape/shapeConst*
dtype0*
valueB"?????  
?
$MobilenetEdgeTPU/Predictions/ReshapeReshapeMobilenetEdgeTPU/Logits/output*MobilenetEdgeTPU/Predictions/Reshape/shape*
T0*
Tshape0
^
$MobilenetEdgeTPU/Predictions/SoftmaxSoftmax$MobilenetEdgeTPU/Predictions/Reshape*
T0
d
"MobilenetEdgeTPU/Predictions/ShapeShapeMobilenetEdgeTPU/Logits/output*
T0*
out_type0
?
&MobilenetEdgeTPU/Predictions/Reshape_1Reshape$MobilenetEdgeTPU/Predictions/Softmax"MobilenetEdgeTPU/Predictions/Shape*
Tshape0*
T0"?