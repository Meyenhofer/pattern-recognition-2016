#!/usr/bin/env ruby

require 'csv'

def sort_kws_csv(input, output)
  hash = read_csv(input)
  sorted = []
  hash.each do |key, values|
    row = [key]
    row << values.sort_by { |_k, v| v }
    sorted << row
  end
  write_csv(sorted, output)
end

def read_csv(file)
  hash = {}
  CSV.foreach(file) do |row|
    word_label = row.shift.strip
    hash[word_label] = {}
    row.each_slice(2) do |id, dist|
      hash[word_label][id.strip] = dist.strip.to_f
    end
  end
  hash
end

def write_csv(arr, file)
  CSV.open(file, 'wb') do |csv|
    arr.each { |row| csv << row.flatten }
  end
end

if __FILE__ == $PROGRAM_NAME
  case ARGV.length
  when 0
    puts "Usage: #{$PROGRAM_NAME} input.csv output.csv"
    exit 0
  when 1
    sort_kws_csv(ARGV[0], 'kws-sorted.csv')
  else
    sort_kws_csv(ARGV[0], ARGV[1])
  end
end
