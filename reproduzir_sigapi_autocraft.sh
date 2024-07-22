#!/bin/bash
CHECK_PIP=$(which pip)
[ "$CHECK_PIP" != "" ] || { echo "instale o pip: sudo apt -y install python3-pip"; exit; }
PKGS=(pandas numpy scikit-learn)
CHECK_PKGS=`pip show ${PKGS[@]} | grep -i -w "not found"`
[ "$CHECK_PKGS" = "" ] || { echo "instale os pacotes Python: sudo pip install ${PKGS[@]}"; exit; }

set_increment(){
    TOTAL_FEATURES=$1
   # [[ $TOTAL_FEATURES -lt 50 ]] && INCREMENT=1 && return
   # [[ $TOTAL_FEATURES -lt 1000 ]] && INCREMENT=1 && return
    INCREMENT=$((TOTAL_FEATURES * 1 /100))
    echo $INCREMENT
}

sigapi(){
    DATASET=$1
    D_NAME=$(basename "$DATASET" .csv)
    set_increment `head -1 $DATASET | awk -F, '{print NF-1}'`
    echo "python3 -m SigAPI_Otimizado.metodos.SigAPI.main -d $DATASET -o resultado-selecao-BALANCEADOS-$D_NAME.csv -i $INCREMENT"
    python3 -m SigAPI_Otimizado.metodos.SigAPI.main -d $DATASET -o resultado-selecao-BALANCEADOS-$D_NAME.csv -i $INCREMENT
}

for DATASET in Datasets/Balanceados/*.csv
do
    D_NAME=$(basename "$DATASET" .csv)
    TS=$(date +%Y%m%d%H%M%S)
    { time sigapi "$DATASET" "$D_NAME"; } 2> time-sigapi-"$D_NAME"-"$TS".txt
done
