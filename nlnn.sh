PRE_INPUT='./orlab'
PRE_W_PATH='Models/orlab_before'
PRE_Y_PATH='./estlab'
source ~/evdb_py34/bin/activate
function NN {
                    python3.4 ~/bilty/src/bilty.py  $1 
                }
NN $INPUT $W_PATH $Y_PATH


