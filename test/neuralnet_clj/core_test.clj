(ns neuralnet-clj.core-test
  (:use clojure.core.matrix)
  (:require [clojure.test :refer :all]
            [neuralnet-clj.core :refer :all])
  (:gen-class))

;; (defn abs [x]
;;  (if (< x 0)
;;    (- x)
;;    x))

(defn approx=
  "Tests two values for approximity within 1e10^-10."
  [x y]
  (> 0.0000000001 (abs (- x y))))

(defn coll-approx=
  "Tests two collections of values for approximity within 1e10^-10."
  [xcoll ycoll]
  (if (= (count (filterv #(= % true) (map approx= xcoll ycoll)))
         (count xcoll)
         (count ycoll))
    true
    false))

(deftest sigmoid-test
  (testing "Testing sigmoid function:"
    (is (approx= (sigmoid 1) 0.73105857863000487925))
    (is (approx= (sigmoid 2) 0.88079707797788244406))
    (is (approx= (sigmoid 3) 0.95257412682243321912))
    (is (coll-approx= [(sigmoid 1)
                       (sigmoid 2)
                       (sigmoid 3)]
                      [0.73105857863000487925
                       0.88079707797788244406
                       0.95257412682243321912]))
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
      (is (= (ecount rrow) 3))
      (is (between first 0.0 1.0))
      (is (between second 0.0 1.0))
      (is (between third 0.0 1.0))
    )))

(deftest randomize-layer-test
  (testing "Testing randomizing full layer:"
    (let [rmat (randomize-layer (matrix [[1.0 1.0 1.0],
                                         [1.0 1.0 1.0]]))]
      (is (= (row-count rmat) 2))
      (is (= (column-count rmat) 3))
      (doseq [row rmat]
        (doseq [weight row]
          (is (between weight 0.0 1.0))))
    )))

(deftest randomize-net-test
  (testing "Testing randomizing whole network:"
    (let [rnet (randomize-net (neural-net 0 1 2 1 2 1 0 1))
          rweights (:weights rnet)]
      (is (= (row-count (rweights 0)) 2))
      (is (= (column-count (rweights 0)) 2))
      (is (= (row-count (rweights 1)) 2))
      (is (= (column-count (rweights 1)) 2))
      (is (= (row-count (rweights 2)) 2))
      (is (= (column-count (rweights 2)) 1))
      (doseq [layer rweights]
        (doseq [row layer]
          (doseq [weight row]
            (is (between weight 0.0 1.0)))))
    )))

(deftest activate-neuron-test
  (testing "Testing neuron activation function:"
    (is (approx= (activate-neuron [1.0 2.0] [2.0 3.5])
                 0.99995460213129756561))
    (is (approx= (activate-neuron [0.5 0.7 0.3] [0.6 0.9 0.2])
                 0.8797431375322491893))
    (is (approx= (activate-neuron [0.3 1.4 0.1 0.22] [0.04 2.9 1.25 0.78])
                 0.99536096924906201644))
    ))

(deftest activate-layer-test
  (testing "Testing layer activation function:"
    (is (coll-approx= (activate-layer [0.3 2.6]
                                      (matrix [[1.0 2.0],
                                               [2.0 3.5]]))
                      [0.998498817743263
                       0.9999774555703496]))
    (is (coll-approx= (activate-layer [1.5 1.9]
                                      (matrix [[0.5 0.7 0.3],
                                               [0.6 0.9 0.2]]))
                      [0.947349881564697
                       0.9772460565371959
                       0.861761726827506]))
    (is (coll-approx= (activate-layer [3.0 0.01 0.41]
                                      (matrix [[0.3 1.4 0.1]
                                               [0.04 2.9 1.25]
                                               [0.78 0.22 1.1]]))
                      [0.9020488684649232
                       0.9951271910406644
                       0.8536474687936986]))
    ))

(deftest feed-forward-test
  (testing "Testing feeding inputs through a whole network:"
    (let [anet {:input-lo 0
                :input-hi 1
                :num-inputs 2
                :num-hidden-layers 1
                :num-layer-nodes 2
                :num-outputs 1
                :output-lo 0
                :output-hi 1
                :weights [(matrix [[0.3 0.9],
                                   [0.87 0.77]]),
                          (matrix [[0.12 0.43],
                                   [0.39 0.71]]),
                          (matrix [[0.22],
                                   [0.46]])]}]
      (is (coll-approx= (feed-forward [0.3 2.6] anet)
                      [0.8305297476011987])))
    ))

