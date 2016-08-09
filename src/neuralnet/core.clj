(ns neuralnet.core
  (:use clojure.core.matrix)
  (:require [clojure.math.numeric-tower :as math]))

(set-current-implementation :vectorz)

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

;(defn layer-weights [num-inputs num-targets]
  ;(->> (repeat num-inputs 1.0)
       ;(repeat num-targets)
       ;(matrix)))

(defn layer-weights [num-inputs num-targets]
  (->> (repeat num-inputs 1.0)
       (vec)
       (repeat num-targets)
       (vec)))

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

(defn random-weight-net [net]
  (assoc net :weights
         (vec (map (fn [layer]
                     (vec (map (fn [row]
                                 (vec (map (fn [w]
                                             (* (rand) 1.0))
                                           row)))
                               layer)))
                   (:weights net)))))

(defn activate [inputs weights-to-neuron]
  (->> (map * inputs weights-to-neuron)
       (conj* 1.0)
       (reduce +)
       (sigmoid)))

(defn activate-layer [inputs layer]
  (for [weights-to-neuron layer]
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

  ;(->> (:weights net)))

; (def e Math/E)
;
; (defn exp
;   ([base exponent] (exp base exponent 1))
;   ([base times-left accumulator]
;     (if (zero? times-left)
;       accumulator
;       (if (> times-left 0)
;         (recur base (dec times-left) (* accumulator base))
;         (recur base (inc times-left) (/ accumulator base))))))

;(defn abs [n]
  ;(max n (- n)))

;(defn neural-net [num-inputs num-hidden-layers num-layer-nodes num-outputs bias]
  ;(->> (repeat num-layer-nodes 1.0)
       ;(vec)
       ;(repeat num-layer-nodes)
       ;(vec)
       ;(repeat (dec num-hidden-layers))
       ;(vec)
       ;(into [(->> (repeat num-inputs 1.0)
                   ;(vec)
                   ;(repeat num-layer-nodes)
                   ;(vec))])
       ;(conj* (->> (repeat num-layer-nodes 1.0)
                   ;(vec)
                   ;(repeat num-outputs)
                   ;(vec)))
       ;(hash-map :num-inputs num-inputs
                 ;:num-hidden-layers num-hidden-layers
                 ;:num-layer-nodes num-layer-nodes
                 ;:num-outputs num-outputs
                 ;:bias bias
                 ;:weights)))

;(def nn
  ;{:num-inputs 2
   ;:num-hidden-layers 2
   ;:num-outputs 1
   ;:weights [;all layers
             ;[;layer 0
              ;[;weights from layer 0 output 1 to layer 1
               ;w011
               ;w012
              ;]
              ;[;weights from layer 0 output 2 to layer 1
               ;w021
               ;w022
              ;]
             ;]
             ;[;layer 2
              ;[;weights from layer 1 output 1 to layer 2
               ;w211
               ;w212
              ;]
              ;[;weights from layer 1 output 1 to layer 2
               ;w221
               ;w222
              ;]
             ;]
             ;[;output layer
              ;[;weights for single output neuron
               ;w011
               ;w021
              ;]
             ;]
            ;]})

;(for [layer (:weights nn)]
  ;(for [output layer]
    ;(for [target output]
      ;;this weight goes to target in layer from previous output
      ;)))

;(defn percept-layer [input-vals input-weights activation-function]
  ;)

;(defn percept [input-val input-weights activation-function])
