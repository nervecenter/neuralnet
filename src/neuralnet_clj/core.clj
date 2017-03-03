(ns neuralnet-clj.core
  (:use clojure.core.matrix)
  (:require [clojure.math.numeric-tower :as math])
  (:gen-class))

(set-current-implementation :vectorz)

;;          EXAMPLE LAYER:
;
;              Weights
;                 to
;               neuron:
;              a   b   c
;            _____________
;            |   |   |   |
;          1 | w | w | w |
;  Weights   |   |   |   |
;  from    2 | w | w | w |
;  input:    |   |   |   |
;          3 | w | w | w |
;            |   |   |   |
;            -------------
;


;
; GENERAL STUFF
;

(def bias 1.0)
(def learning-rate 0.3)

(defn sigmoid
  "The sigmoid function: S(t) = 1 / (1 + e^-t)"
  [z]
  (/ 1 (+ 1 (math/expt Math/E (- z)))))

(defn conj*
  "Same as conj, but element is first param and collection is second."
  [element coll]
  (conj coll element))

(defn normalize [input-lo input-hi number output-lo output-hi]
  (float
    (+ (/ (* (- number
                input-lo)
             (- output-hi
                output-lo))
          (- input-hi
             input-lo))
       output-lo)))

;
; NETWORK CREATION FUNCTIONS
;

(defn layer-weights
  "Generate a matrix of weights representing a layer by
   providing the number of inputs and outputs, which
   correspond to rows and columns respectively."
  [num-inputs num-targets]
  (->> (repeat num-targets 1.0)
       (repeat num-inputs)
       (matrix)))

(defn neural-net
  "Generate a neural net map, containing relevant figures
   to verify the net and a sequence of weight matrices
   representing the net itself."
  [input-lo
   input-hi
   num-inputs
   num-hidden-layers
   num-layer-nodes
   num-outputs
   output-lo
   output-hi]
  (->> (layer-weights num-layer-nodes num-layer-nodes)
       (repeat num-hidden-layers)
       (vec)
       (into [(layer-weights num-inputs num-layer-nodes)])
       (conj* (layer-weights num-layer-nodes num-outputs))
       (hash-map :input-lo input-lo
                 :input-hi input-hi
                 :num-inputs num-inputs
                 :num-hidden-layers num-hidden-layers
                 :num-layer-nodes num-layer-nodes
                 :num-outputs num-outputs
                 :output-lo output-lo
                 :output-hi output-hi
                 :weights)))

(defn randomize-row
  "Take a row from a layer matrix and map random weights to it."
  [row]
  (->> row
       (map (fn [weight] (* (rand) 1.0)))
       (array)))

(defn randomize-layer
  "Take a layer matrix and map it with random weights."
  [layer]
  (->> layer
       (map #(randomize-row %))
       (matrix)))

(defn randomize-net
  "Take a neural net and randomize the sequence of weight matrices."
  [net]
  (->> (:weights net)
       (map #(randomize-layer %))
       (vec)
       (assoc net :weights)))

;
; UTILIZATION FUNCTIONS
;

(defn activate-neuron
  "Dot product the inputs to a neuron with the respective
   weights of their connections, and then apply the sigmoid
   to get the activation value, or output, of the neuron."
  [inputs weights-to-neuron]
  (->> (map * inputs weights-to-neuron)
       (conj* 1.0)
       (reduce +)
       (sigmoid)))

(defn activate-layer
  "Takes a set of inputs from the previous layer and a layer
   matrix and activates all the neurons, producing a set of
   outputs to feed forward to the next layer."
  [inputs layer]
  (for [weights-to-neuron (columns layer)]
    (activate-neuron inputs weights-to-neuron)))

;; (defn feed-forward
;;   "Take a set of inputs and completely feed them through a
;;    given network, producing the corresponding outputs."
;;   [inputs network]
;;   (loop [prev-vals (vec
;;                      (map #(normalize (:input-lo network)
;;                                       (:input-hi network)
;;                                       %
;;                                       -1.0
;;                                       1.0)
;;                           inputs))
;;          layers-remaining (:weights network)]
;;     (if (empty? layers-remaining)
;;       (vec (map #(normalize -1.0
;;                             1.0
;;                             %
;;                             (:output-lo network)
;;                             (:output-hi network))
;;                 prev-vals))
;;       (recur (activate-layer prev-vals (first layers-remaining))
;;              (rest layers-remaining)))))

;
; TRAINING FUNCTIONS
;

;; (defn mean-squared-error
;;   "Formula for calculating the error of the output of a
;;    network. Not for hidden layers."
;;   [actual expected]
;;   (* learning-rate (math/expt (- actual expected) 2)))

;; (defn neuron-error
;;   "Takes the previous errors and dot products them with the
;;    weights of the given hidden neuron, calculating its error."
;;   [prev-errors neuron-weights]
;;   (dot prev-errors neuron-weights))

;; (defn layer-errors
;;   "Calculate the errors for the neurons in a layer by
;;    taking the dot product of the following layer's errors
;;    and the weights of the current layer."
;;   [prev-errors layer]
;;   (vec (for [neuron-weights (rows layer)]
;;          (neuron-error prev-errors neuron-weights))))

;; (defn adjust-column-weights
;;   "Add the errors to the current weights to a neuron to
;;    produce newly adjusted weights."
;;   [neuron-error weights]
;;   (map #(+ neuron-error %) weights))

;; (defn adjust-layer-weights
;;   "Produce a new layer by applying the errors at that layer
;;    to the current layer."
;;   [layer-errors layer]
;;   (array (map #(adjust-column-weights %1 %2) layer-errors (columns layer))))

;; (defn adjust-net-weights
;;   "Produce a new sequence of layer matrices by adjusting
;;    layer weights based on the errors at each layer."
;;   [errors network]
;;   (->> (map #(adjust-layer-weights %1 %2) errors (:weights network))
;;        (assoc network :weights)))

;; (defn back-propagate
;;   "Feed forward test data. Based on the output, calculate
;;    errors, then propagate those errors back and return an
;;    adjusted network."
;;   [inputs expected-output network]
;;   (loop [errors (->> (map #(mean-squared-error %1 %2)
;;                           (feed-forward inputs network)
;;                           expected-output)
;;                      (vec)
;;                      (list))
;;          layers (reverse (:weights network))]
;;     (println "Error:" (first errors))
;;     (if (empty? layers)
;;       (adjust-net-weights (vec errors) network)
;;       (recur (conj errors
;;                    (layer-errors (first errors) (first layers)))
;;              (rest layers)))))

;; (defn epoch
;;   "Back propagate each piece of data (inputs and their
;;    expected outputs) in a sequence of test data, returning
;;    the adjusted network."
;;   [training-data network]
;;   (loop [data training-data
;;          net network]
;;     (if (empty? data)
;;       net
;;       (recur (rest data)
;;              (back-propagate (:in (first data)) (:out (first data)) net)))))

;; (defn train-network
;;   "Train a network using the given training data, halting
;;    at the desired error."
;;   [network training-data desired-error]
;;   (loop [epochs 1]
;;     (recur (inc epochs))))

;
; TEST SCENARIO
;

;; (def and-gate-training-data
;;   [{:in  [0.0 0.0]
;;     :out [0.0]}
;;    {:in  [1.0 0.0]
;;     :out [0.0]}
;;    {:in  [0.0 1.0]
;;     :out [0.0]}
;;    {:in  [1.0 1.0]
;;     :out [1.0]}])

;; (def and-gate-network
;;   (-> (neural-net 0.0 1.0 2 2 2 1 0.0 1.0)
;;       (randomize-net)
;;       (train-network and-gate-training-data 0.05)))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))

; (defn exp
;   ([base exponent] (exp base exponent 1))
;   ([base times-left accumulator]
;     (if (zero? times-left)
;       accumulator
;       (if (> times-left 0)
;         (recur base (dec times-left) (* accumulator base))
;         (recur base (inc times-left) (/ accumulator base))))))
