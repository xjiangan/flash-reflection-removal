
cd data/flashAmbient

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