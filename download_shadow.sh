
cd data

file="SynShadow.zip"
if [[ ! -e "$file" ]];then
    wget "http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/SynShadow.zip"
fi

if unzip -nq "$file"; then
    rm "$file"
fi


cd ../