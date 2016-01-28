# Generates a list of commands to train RBMs, with different parameters
# and training sets.
#
# Copyright (c) 2016 Gijs van Tulder / Erasmus MC, the Netherlands
# This code is licensed under the MIT license. See LICENSE for details.

# somewhat balanced folds
folds = [
  [35, 78, 83, 94, 101, 120, 134, 135, 149, 157, 159, 167, 169, 171, 172],
  [7, 47, 65, 74, 77, 82, 89, 92, 105, 138, 155, 158, 175, 182, 183],
  [3, 23, 39, 56, 62, 76, 107, 112, 118, 126, 131, 147, 153, 160, 180],
  [17, 19, 45, 48, 51, 84, 109, 122, 127, 132, 136, 140, 152, 185],
  [36, 37, 80, 81, 90, 116, 121, 124, 143, 144, 163, 164, 166, 168]
]

fold_file_lists = folds.map do |fold|
  fold.map do |scan_id|
    "ILD-cells/#{ scan_id }.mat"
  end.join(",")
end

folds.size.times do |test_fold|
  train_scans = []
  test_scans = []
  fold_file_lists.each_with_index do |fold_file_list, fold|
    if fold == test_fold
      test_scans << fold_file_list
    else
      train_scans << fold_file_list
    end
  end
  train_scans = train_scans.join(",")
  test_scans = test_scans.join(",")

  exp = []

  [ 5, 8, 10 ].each do |filter_size|
    [ 4, 16, 25, 36 ].each do |filter_count|
      File.open("recipes/conv-#{ filter_size }x#{ filter_size }x#{ filter_count }", "a") do |f|
        [ "--ignore-labels",
          "--beta 1.0",
          "--beta 0.001",
          "--beta 0.01",
          "--beta 0.1",
          "--beta 0.2",
          "--beta 0" ].each do |beta|
          %w{ 0.000001 }.each do |w_init|
            %w{ 0.001 0.00001 0.0000001 0.00000001 0.000000001 }.each do |learning_rate|
              if test_fold > 0
                seed = 123 + 2 * test_fold
                seed = "--rng-seed #{ seed } "
              else
                seed = ""
              end
              experiment_id = "exp-testfold#{ test_fold }-normPATCH-conv-#{ filter_size }x#{ filter_size }x#{ filter_count }-beta#{ beta.gsub(/[^.0-9]+/,"") }-winit#{ w_init }-lrate#{ learning_rate }"
              f.puts "python -u exp_train_rbm.py --epochs 1001 --learning-rate #{ learning_rate } --filter-height #{ filter_size } --filter-width #{ filter_size } --image-size 32 --hidden-maps #{ filter_count } #{ beta } --beta-decay 1 --k-eval 1 --mb-size 5 --n-states 5 #{ seed }--train-scans #{ train_scans } --test-scans #{ test_scans } --convolution-type full --weight-w-init-std #{ w_init } --weight-u-init-std #{ w_init } --evaluate-every 100 --plot-every 100 --test-every 100 --experiment-id #{ experiment_id }"
              # removed: --global-normalisation 
            end
          end
        end
      end
    end
  end
end

