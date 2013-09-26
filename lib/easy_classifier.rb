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
    end
end