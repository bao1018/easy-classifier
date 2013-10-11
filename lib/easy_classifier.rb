require 'easy_classifier/version'
require 'nbayes'
require 'rb-libsvm'

module Classifier
	class API
		def initialize(options = {})

			@path = options[:path] || './corpus/'
			@nbayes = options[:@nbayes] || true
			@nsvm = options[:nsvm] || true

		end

		private

		def train_nbayes(category_corpus, category_name)
			bayes_model_file_name = @path + category_name + "_bayes.yml"
			nbayes = NBayes::Base.new
			category_corpus['data'].values.each do |value|
				nbayes.train(value['text'].split(/\s+/), value['category'])
			end

			nbayes.dump(bayes_model_file_name)
			debug "NBayes Model is Updated"

		end

  #nbayes training logic
  def nbayes_training(category_corpus, category_name)
  	bayes_model_file_name = CAT_BASE_PATH + category_name + "_bayes.yml"
  	
  	nbayes = NBayes::Base.new
  	category_corpus['data'].values.each do |value|
  		nbayes.train(value['text'].split(/\s+/), value['category'])
  	end

  	nbayes.dump(bayes_model_file_name)
  	debug "NBayes Model is Updated"
  end
  #svm training logic
  def svm_training(category_corpus, category_name)
  	svm_model_file_name = CAT_BASE_PATH + category_name + "_svm.yml"
  	svm_dict_file_name = CAT_BASE_PATH + category_name + "_svm_dict.yml"
    #format corpus before model training, all SVM training needed is [[1, text1], [2, text2],...]
    docs = category_corpus['data'].values.map do |row|
    	row = row.values
    	row = row.drop 1
    	row[0] = row[0].to_i
    	row
    end
    
    #train SVM dictionary
    dictionary = docs.map(&:last).map(&:split).flatten
    #train SVM model
    training_set = []
    docs.each do |doc|
    	features_array = dictionary.map { |x| doc.last.include?(x) ? 1 : 0 }
    	training_set << [doc.first, Libsvm::Node.features(features_array)]
    end
    problem = Libsvm::Problem.new
    parameter = Libsvm::SvmParameter.new

    parameter.cache_size = 5 # in megabytes
    parameter.eps = 0.001
    parameter.c = 10

    problem.set_examples(training_set.map(&:first), training_set.map(&:last))
    
    model = Libsvm::Model.train(problem, parameter)
    model.save(svm_model_file_name)
    # #store SVM dictionary
    File.open(svm_dict_file_name, 'w' ) do |out|
    	YAML.dump(dictionary, out)
    end
    debug "SVM Model & Dictionary are Updated"
end
end
end