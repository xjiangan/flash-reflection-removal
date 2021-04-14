
data_root="data"

if [[! -e "$data_root" ]];then 
    mkdir "$data_root"
fi

cd "$data_root"

if [[ ! -e "SynShadow" ]];then
    file="SynShadow.zip"
    if [[ ! -e "$file" ]];then
        wget "http://www.hal.t.u-tokyo.ac.jp/~inoue/projects/synthetic_shadow/SynShadow.zip"
    fi

    if unzip -nq "$file"; then
        rm "$file"
    fi
fi

if [[ ! -e "flashAmbient" ]];then
    mkdir "flashAmbient"
fi

cd flashAmbient

categories=(
    "People"
    "Shelves"
    "Plants"
    "Toys"
    "Rooms"
    "Objects"
)

echo categories: ${categories[@]}

for category in "${categories[@]}"; do

    name="${category}_Photos"
    if [[ -e "$name" ]]; then
        echo "skip $name"
        continue
    fi
    
    file="${name}.zip"
    link="https://cvg.ethz.ch/research/flash-ambient/${file}"

    if [[ ! -e "$file" ]];then
        wget "${link}"
    fi

    if unzip -nq "${file}"; then
        rm "${file}"
    fi
done
cd ../..
