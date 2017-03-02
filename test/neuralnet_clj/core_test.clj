(ns neuralnet-clj.core-test
  (:use clojure.core.matrix)
  (:require [clojure.test :refer :all]
            [neuralnet-clj.core :refer :all])
	(:gen-class))

;; (defn abs [x]
;; 	(if (< x 0)
;; 		(- x)
;; 		x))

(defn approx= [x y]
	(> 0.0000000001 (abs (- x y))))

(deftest sigmoid-test
	(testing "Testing sigmoid function:"
		(is (approx= (sigmoid 1) 0.73105857863000487925))
		(is (approx= (sigmoid 2) 0.88079707797788244406))
		(is (approx= (sigmoid 3) 0.95257412682243321912))
		))

(deftest conj*-test
	(testing "Testing appending items with conj*:"
		(is (= (conj* 3 [1 2]) [1 2 3]))
		(is (= (conj* 4 [2 3]) [2 3 4]))
		(is (= (conj* 5 [3 4]) [3 4 5]))
		))

(deftest normalize-test
	(testing "Testing normalizing numbers to new range:"
		(is (= (normalize 0.0 1.0 0.5 0.0 10.0) 5.0))
		(is (= (normalize 0.0 1.0 0.5 -10.0 10.0) 0.0))
		(is (= (normalize -1.0 1.0 0.5 0.0 10.0) 7.5))
		))

(deftest layer-weights-test
	(testing "Testing creation of layers of weights:"
		(is (= (layer-weights 2 2)
					 (matrix [[1, 1],
										[1, 1]])))
		(is (= (layer-weights 2 3)
					 (matrix [[1, 1, 1],
										[1, 1, 1]])))
		(is (= (layer-weights 3 2)
					 (matrix [[1, 1],
										[1, 1],
										[1, 1]])))
		))

(deftest neural-net-test
	(testing "Testing creation of a prototype neural net:"
		(is (= (neural-net 0 1 2 1 2 1 0 1)
					 {:input-lo 0
						:input-hi 1
						:num-inputs 2
						:num-hidden-layers 1
						:num-layer-nodes 2
						:num-outputs 1
						:output-lo 0
						:output-hi 1
						:weights [(matrix [[1, 1],
															 [1, 1]]),
											(matrix [[1, 1],
															 [1, 1]]),
											(matrix [[1],
															 [1]])]}))
		))

(defn between [num lo hi]
	(if (and (<= lo num) (<= num hi)) true false))

(deftest randomize-row-test
	(testing "Testing randomizing layer rows:"
		(let [rrow (randomize-row [1.0 1.0 1.0])
					first (select rrow 0)
					second (select rrow 1)
					third (select rrow 2)]
			(is (between first 0.0 1.0))
			(is (between second 0.0 1.0))
			(is (between third 0.0 1.0))
		)))

(deftest randomize-layer-test
	(testing "Testing randomizing whole layer:"
		(let [rmat (randomize-layer (matrix [[1.0 1.0 1.0],
																				 [1.0 1.0 1.0]]))]
			(doseq [row rmat]
				(doseq [weight row]
					(is (between weight 0.0 1.0))))
		)))
