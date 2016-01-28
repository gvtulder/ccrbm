#!/bin/bash
# Runs a command and writes the output to a file, but only if this file
# does not already exist.
#
# Usage:
#   ./run-with-output.sh output_file command <argument 1> ...
#
# This script will first write the output to a temporary file. If the
# command is successful, the temporary file is moved to the destination.
#
# Trailing arguments are passed to the command.
#
# Copyright (c) 2016 Gijs van Tulder / Erasmus MC, the Netherlands
# This code is licensed under the MIT license. See LICENSE for details.
set -e
output_file=$1
shift

mkdir -p $( dirname $output_file )
if [ -f $output_file ] ; then
  echo "Already exists: $output_file"
  exit
fi

if [[ -z $TMPDIR ]] ; then
  tmp_file=/tmp/output-eval-$$
else
  tmp_file=$TMPDIR/output-eval-$$
fi

rm -rf $tmp_file

echo $@ >> $tmp_file
$@ >> $tmp_file

result=$?

if [[ $result == 0 ]] ; then
  mv $tmp_file $output_file
  exit $result
else
  rm $tmp_file
  exit $result
fi

