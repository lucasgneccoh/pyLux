# First run the main script
output_file=$1
to=$2
pwd=$(pwd)
# Run the script
python3 expert_iteration_external.py --inputs ../support/exp_iter_inputs/exp_iter_inputs_small.json --verbose 0 > $output_file
# Send email
touch tmp_mail.sh
echo 'echo "Automatic mail" | mail -s "Expert iteration finished" -A '$pwd'/'$output_file' '$to > tmp_mail.sh
ssh -p 5022 lgnecco@lamgate4 'bash -s' < tmp_mail.sh
rm tmp_mail.sh
echo "Done"
