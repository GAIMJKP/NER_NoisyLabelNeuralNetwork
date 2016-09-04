#!/bin/bash

# init
export LD_LIBRARY_PATH=/nethome/evdberg/cnn/pycnn
echo $LD_LIBRARY_PATH

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8

#source /nethome/evdberg/evdb_py34/bin/activate 
python --version

NOISELEV=$1
ESTTRAIN_XYC='NER_estimatedlabels/noisy'$NOISELEV'_eng.train_xyc'
ESTTRAIN_XC='NER_estimatedlabels/noisy'$NOISELEV'_eng.train_xc'
MODEL_PATH='Models/'$NOISELEV'after'

cat $ESTTRAIN_XYC | head -n 10
# first_time: save theta to check convergence, then update c and theta based on labels estimated by pretrained model
python singleEM.py -noise $NOISELEV -initialize
# evaluate bsed on one EM it to have something to determine convergence on
perl conlleval.pl -d '\t' < $ESTTRAIN_XYC  > conlleval_before_one_it
cat $ESTTRAIN_XYC | head -n 10

# enter loop, check for convergence
ITER=0
MAXITER=15
state=$(cat state.txt)

if test "$state" != "Converged"

then

while [ "$ITER" -lt "$MAXITER" ];

do echo $ITER;
cat $ESTTRAIN_XYC | head -n 10;
python /nethome/evdberg/bilstm-aux/src/bilty.py --train $ESTTRAIN_XC --test test.txt --output $ESTTRAIN_XYC --pred_layer 1 --iters 30 --h_dim 50 --save $MODEL_PATH;
ITER=$((ITER+1));
cat $ESTTRAIN_XYC | head -n 10;
python singleEM.py -noise $NOISELEV;

perl conlleval.pl -d '\t' < $ESTTRAIN_XYC  > conlleval_after_one_it;
python compare_f1score.py;
cp conlleval_after_one_it conlleval_before_one_it;

done

else

echo 'Converged'

fi
