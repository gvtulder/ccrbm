# This script schedules the random forest classification experiments.
#
# For each of the trained RBMs, it trains a number of random forests
# with various parameters.
#
# Copyright (c) 2016 Gijs van Tulder / Erasmus MC, the Netherlands
# This code is licensed under the MIT license. See LICENSE for details.
require "fileutils"

# somewhat balanced folds
FOLDS = [
  [35, 78, 83, 94, 101, 120, 134, 135, 149, 157, 159, 167, 169, 171, 172],
  [7, 47, 65, 74, 77, 82, 89, 92, 105, 138, 155, 158, 175, 182, 183],
  [3, 23, 39, 56, 62, 76, 107, 112, 118, 126, 131, 147, 153, 160, 180],
  [17, 19, 45, 48, 51, 84, 109, 122, 127, 132, 136, 140, 152, 185],
  [36, 37, 80, 81, 90, 116, 121, 124, 143, 144, 163, 164, 166, 168]
]


#
# for one RBM
#  1. generate feature vectors 
#  2. train and evaluate RFs
#  3. remove feature vectors
#

CACHE_DIR = "cache/#{ Time.now.strftime("%Y%m%d-%H%M%S") }"

def wait_for_empty_slot
  while Dir["cache/*"].size > 30
    puts "Waiting..."
    sleep 60
  end
end


def scan_ids_to_str(scan_ids)
  scan_ids.map{|scan_id|"ILD-cells/#{ scan_id }.mat"}.join(",")
end

def bigrsub(args)
  q_cmd = "bigrsub -l eval-logs/ #{ args }"
  IO.popen(q_cmd) { |io| io.read }.strip
end

def schedule_rf_tasks(pkl_file, rbm_id, fold_id, train_scan_ids, test_scan_ids, feature_seed, fold_cache_dir)
  wait_for_empty_slot

  # schedule classifier tasks, if the output file does not exist
  eval_tasks = []
  %w{ conv.bins-2 conv.bins-4 conv.bins-8 }.each do |histogram_config|
    number_of_features_available = (histogram_config[/[0-9]+/].to_i * rbm_id[/[0-9]+x[0-9]+x([0-9]+)/,1].to_i)

    [10,20,50,100,200].each do |n_estimators|
      eval_tasks_subgroup = []

      [1,2,4,8,16,32,48,64,96,128,256,number_of_features_available].uniq.each do |max_features|
        next if max_features > number_of_features_available

        [123,345,567].each do |rf_seed|
          output_file = "rf-results/#{ rbm_id }/#{ fold_id }/#{ histogram_config }-n_estimators-#{ n_estimators }-max_features-#{ max_features}-rf_seed-#{ rf_seed }"
          predictions_file = "rf-results/#{ rbm_id }/#{ fold_id }/#{ histogram_config }-n_estimators-#{ n_estimators }-max_features-#{ max_features}-rf_seed-#{ rf_seed }-predictions.npz"

          # step 2: train and evaluate classifiers
          cmd = %{ ./run-with-output.sh #{ output_file }
               python -u experiment_random_forest.py
                      --experiment-id q
                      --train-set #{ fold_cache_dir }/#{ histogram_config }/data.train.npz
                      --test-set #{ fold_cache_dir }/#{ histogram_config }/data.test.npz
                      --save-predictions #{ predictions_file }
                      --n-estimators #{ n_estimators }
                      --max-features #{ max_features }
                      --seed #{ rf_seed } }.gsub(/\s+/, " ").strip

          eval_tasks_subgroup << cmd unless File.exists?(output_file)
        end
      end

      if not eval_tasks_subgroup.empty?
        eval_tasks << "bash -c '#{ eval_tasks_subgroup.join(" ; ") }'"
      end
    end
  end

  # exit if all tasks for these settings are done
  return if eval_tasks.empty?

  FileUtils.mkdir_p(fold_cache_dir)

  # step 1: generate feature vectors
  cmd = %{
       python -u exp_save_features.py
              --previous-layer #{ pkl_file }
              #{ pkl_file=~/random-filters/ ? "--random-filters --random-filters-seed #{ feature_seed }" : "" }
              --train-scans #{ scan_ids_to_str(train_scan_ids) }
              --test-scans #{ scan_ids_to_str(test_scan_ids) }
              --n-states 5
              --rng-seed 123
              --convolution-type full
              --pooling-approach histograms
              --skip-sigmoid
              --save-features #{ fold_cache_dir } }.gsub(/\s+/, " ").strip
# system(cmd)
  job_id_save = bigrsub("-q day,week,month -c 8 -R 10G -N ILD-eval-step-1 #{ cmd }")
# system("qalter -h u #{ job_id_save }")
# job_id_save = 0

  # step 2: train and evaluate classifiers
  job_ids_eval = []
# eval_tasks.each do |cmd|
#   job_ids_eval << bigrsub("-q week,month -R 2G -N ILD-eval-step-2-rf -j #{ job_id_save } #{ cmd }")
# end
  File.open("#{ fold_cache_dir }/eval-tasks.txt", "w") do |f|
    f.puts eval_tasks
  end
  job_ids_eval << bigrsub("-q hour,day,week,month -R 2G -N ILD-eval-step-2-rf -t #{ fold_cache_dir }/eval-tasks.txt -j #{ job_id_save }")

  # step 3: cleanup
  cmd = "rm -rf #{ fold_cache_dir }"
  bigrsub("-q hour -R 350M -N ILD-eval-step-3-clean -j #{ job_ids_eval.join(",") } #{ cmd }")
end


def schedule_validation_and_test_tasks(classifier, pkl_file, rbm_id, feature_seed, test_fold)
  fold_ids = (0...FOLDS.size).to_a

  # validation of classifier parameters (train, validation and test sets)
  (fold_ids-[test_fold]).each do |validation_fold|
    train_folds = (fold_ids-[validation_fold,test_fold])
    train_scan_ids = train_folds.map{|f|FOLDS[f]}.flatten
    validation_scan_ids = FOLDS[validation_fold]

    fold_id = "feature_seed-#{ feature_seed }/test-#{ test_fold }/validation-#{ validation_fold }"
    fold_cache_dir = "#{ CACHE_DIR }-#{ rbm_id }-#{ fold_id.gsub("/","-") }"

    if classifier == :rbm
      schedule_rbm_tasks(pkl_file, rbm_id, fold_id, train_scan_ids, validation_scan_ids, feature_seed, fold_cache_dir)
    else
      schedule_rf_tasks(pkl_file, rbm_id, fold_id, train_scan_ids, validation_scan_ids, feature_seed, fold_cache_dir)
    end
  end

  # test of classifier (train+validation and test set)
  train_folds = (fold_ids-[test_fold])
  train_scan_ids = train_folds.map{|f|FOLDS[f]}.flatten
  test_scan_ids = FOLDS[test_fold]

  fold_id = "feature_seed-#{ feature_seed }/test-#{ test_fold }"
  fold_cache_dir = "#{ CACHE_DIR }-#{ rbm_id }-#{ fold_id.gsub("/","-") }"

  if classifier == :rbm
    schedule_rbm_tasks(pkl_file, rbm_id, fold_id, train_scan_ids, test_scan_ids, feature_seed, fold_cache_dir)
  else
    schedule_rf_tasks(pkl_file, rbm_id, fold_id, train_scan_ids, test_scan_ids, feature_seed, fold_cache_dir)
  end

  print "."
end

def schedule_rbm_tasks(pkl_file, rbm_id, fold_id, train_scan_ids, test_scan_ids, feature_seed, fold_cache_dir)
  # schedule classifier tasks, if the output file does not exist
  output_file = "rbm-results/#{ rbm_id }/#{ fold_id }-result.txt"
  predictions_file = "rbm-results/#{ rbm_id }/#{ fold_id }-predictions.npz"

  # step 2: train and evaluate classifiers
  cmd = %{ ./run-with-output.sh #{ output_file }
       python -u exp_rbm_classification.py
              --previous-layer #{ pkl_file }
              --train-scans #{ scan_ids_to_str(train_scan_ids) }
              --test-scans #{ scan_ids_to_str(test_scan_ids) }
              --save-predictions #{ predictions_file }
              --n-states 5
              --rng-seed 123
              --convolution-type full
              }.gsub(/\s+/, " ").strip

  unless File.exists?(output_file)
    bigrsub("-q day,week,month -R 5G -N ILD-eval-rbm #{ cmd }")
  end
end


# pkl_file = "results/rbm-exp-testfold0-normPATCH-conv-5x5x4-beta-winit0.000001-lrate0.0000001-20141008-130757-58356-epoch-1000.pkl"

Dir["results/*-epoch-1000.pkl"].each do |pkl_file|
  if pkl_file =~ /x25/
    puts "Skipping x25: #{ pkl_file }"
    next
  end

# Dir["results/*201504*-epoch-1000.pkl"].each do |pkl_file|
  rbm_id = File.basename(pkl_file)
  feature_seed = 123
  test_fold = pkl_file[/testfold([0-9]+)/, 1].to_i

  if pkl_file=~/beta[0-9]/
    schedule_validation_and_test_tasks(:rbm, pkl_file, rbm_id, feature_seed, test_fold)
  end
  schedule_validation_and_test_tasks(:rf, pkl_file, rbm_id, feature_seed, test_fold)
end

%w{ filter-banks/filter-bank-LM-32x32x48.pkl
    filter-banks/filter-bank-S-31x31x13.pkl
    filter-banks/filter-bank-LM-16x16x48.pkl
    filter-banks/filter-bank-S-15x15x13.pkl
}.each do |pkl_file|
  rbm_id = File.basename(pkl_file)
  feature_seed = 123
  (0...FOLDS.size).each do |test_fold|
    schedule_validation_and_test_tasks(:rf, pkl_file, rbm_id, feature_seed, test_fold)
  end
end

{  4 => [ 5, 8, 10 ],
  16 => [ 5, 8, 10 ],
  36 => [ 5, 8, 10 ],
  13 => [ 15, 31 ],
  48 => [ 16, 32 ]
}.each do |number_of_filters, filter_sizes|
  filter_sizes.each do |filter_size|
    pkl_file = "results/random-filters-#{ filter_size }x#{ filter_size }x#{ number_of_filters }.pkl"
    [ 123, 456, 789, 321, 654, 987 ].each do |feature_seed|
      (0...FOLDS.size).each do |test_fold|
        rbm_id = File.basename(pkl_file)
        schedule_validation_and_test_tasks(:rf, pkl_file, rbm_id, feature_seed, test_fold)
      end
    end
  end
end

