# First run the main script
output_file=$1
to=$2
#python3 expert_iteration_external.py --inputs ../support/exp_iter_inputs/exp_iter_inputs.json --verbose 0 > $output_file
echo "Testing with a created log file" > $output_file
touch tmp_mail.sh
echo 'mail -s "expert iteration finished" -A '$output_file' '$to > tmp_mail.sh
ssh -p 5022 lgnecco@lamgate4 'bash -s' < tmp_mail.sh
echo "Done"
