(ns neuralnet.core
  (:use clojure.core.matrix)
  (:require [clojure.math.numeric-tower :as math]))

(set-current-implementation :vectorz)

(def bias 1.0)
(def learning-rate 0.3)

(defn sigmoid [z]
  (/ 1 (+ 1 (math/expt Math/E (- z)))))

(defn conj* [element coll]
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

(defn layer-weights [num-inputs num-targets]
  (->> (repeat num-targets 1.0)
       (repeat num-inputs)
       (matrix)))

(defn neural-net [input-lo
                  input-hi
                  num-inputs
                  num-hidden-layers
                  num-layer-nodes
                  num-outputs
                  output-lo
                  output-hi]
  (->> (layer-weights num-layer-nodes num-layer-nodes)
       (repeat (dec num-hidden-layers))
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

(defn randomize-row [row]
  (->> row
       (map (fn [weight] (* (rand) 1.0)))
       (array)))

(defn randomize-layer [layer]
  (->> layer
       (map #(randomize-row %))
       (matrix)))

(defn randomize-net [net]
  (->> (:weights net)
       (map #(randomize-layer %))
       (vec)
       (assoc net :weights)))

(defn activate [inputs weights-to-neuron]
  (->> (map * inputs weights-to-neuron)
       (conj* 1.0)
       (reduce +)
       (sigmoid)))

(defn activate-layer [inputs layer]
  (for [weights-to-neuron (columns layer)]
    (activate inputs weights-to-neuron)))

(defn feed-forward [inputs network]
  (loop [prev-vals (vec
                     (map #(normalize (:input-lo network)
                                      (:input-hi network)
                                      %
                                      -1.0
                                      1.0)
                          inputs))
         layers-remaining (:weights network)]
    (if (empty? layers-remaining)
      (vec (map #(normalize -1.0
                            1.0
                            %
                            (:output-lo network)
                            (:output-hi network))
                prev-vals))
      (recur (activate-layer prev-vals (first layers-remaining))
             (rest layers-remaining)))))

(defn mean-squared-error [actual expected]
  (* 0.01 (math/expt (- actual expected) 2)))

(defn neuron-error
  "Takes a vector of errors from the target neurons and the vector of weights from the desired neuron, and calculates the error by taking the dot product of the two."
  [prev-errors neuron-weights]
  (dot prev-errors neuron-weights))

(defn layer-errors [prev-errors layer]
  (vec (for [neuron-weights (rows layer)]
         (neuron-error prev-errors neuron-weights))))

(defn adjust-column-weights [neuron-error weights]
  (map #(+ neuron-error %) weights))

(defn adjust-layer-weights [layer-errors layer]
  (array (map #(adjust-column-weights %1 %2) layer-errors (columns layer))))

(defn adjust-net-weights [errors network]
  (->> (map #(adjust-layer-weights %1 %2) errors (:weights network))
       (assoc network :weights))

(defn back-propagate [inputs expected-output network]
  (loop [errors (->> (map #(mean-squared-error %1 %2)
                          (feed-forward inputs network)
                          expected-output)
                     (vec)
                     (list))
         layers (reverse (:weights network))]
    ;(println "Errors:")
    ;(doseq [e errors] (println e))
    ;(println "Layers:" layers)
    (if (empty? layers)
      (adjust-net-weights (vec errors) network)
      (recur (conj errors
                   (layer-errors (first errors) (first layers)))
             (rest layers)))))

(defn epoch [training-data network]
  (loop [data training-data
         net network]
    (if (empty? data)
      net
      (recur (rest data)
             (back-propagate (:in (first data)) (:out (first data)) net)))))

(def untrained-and
  (-> (neural-net 0.0 1.0 2 2 2 1 0.0 1.0)
      (randomize-net)))

; (defn exp
;   ([base exponent] (exp base exponent 1))
;   ([base times-left accumulator]
;     (if (zero? times-left)
;       accumulator
;       (if (> times-left 0)
;         (recur base (dec times-left) (* accumulator base))
;         (recur base (inc times-left) (/ accumulator base))))))
