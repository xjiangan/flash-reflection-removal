arr=(bio3 bio4 bio5 bio6 exam2 exam3 exam4)
for name in ${arr[@]} ;
do
echo $name;
rclone copy onedrive:urop/1100/docshadow/$name ./$name
done