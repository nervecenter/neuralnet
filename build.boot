#!/usr/bin/env boot

(set-env! :project 'neuralnet
          :version "0.5.0"
          :source-paths #{"src"}
          :resource-paths #{"resources"}
          :dependencies '[[org.clojure/clojure "1.8.0"]
                          [org.clojure/tools.namespace "0.2.11"]
                          [net.mikera/vectorz-clj "0.44.1"]
                          [org.clojure/math.numeric-tower "0.0.4"]])

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

(use 'clojure.core.matrix)
(require '[clojure.math.numeric-tower :as math])
(set-current-implementation :vectorz)

;(defn abs [n]
  ;(max n (- n)))

(defn sigmoid [z]
  (/ 1 (+ 1 (math/expt Math/E (- z)))))

(defn conj* [element coll]
  (conj coll element))

(defn layer-weights [num-inputs num-targets]
  (->> (repeat num-inputs 1.0)
       (repeat num-targets)
       (matrix)))

(defn neural-net [num-inputs num-hidden-layers num-layer-nodes num-outputs bias]
  (->> (layer-weights num-layer-nodes num-layer-nodes)
       (repeat (dec num-hidden-layers))
       (vec)
       (into [(layer-weights num-inputs num-layer-nodes)])
       (conj* (layer-weights num-layer-nodes num-outputs))
       (hash-map :num-inputs num-inputs
                 :num-hidden-layers num-hidden-layers
                 :num-layer-nodes num-layer-nodes
                 :num-outputs num-outputs
                 :bias bias
                 :weights)))

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

(defn fib
  ([n]
   (fib [0 1] n))
  ([pair, n]
   (print (first pair) " ")
   (if (> n 0)
     (fib [(second pair) (apply + pair)] (- n 1))
     (println))))

(defn -main [& args]
  (let [limit (first args)]
    (println "Printing fibonacci sequence up to " limit "numbers...")
    (fib (Integer/parseInt limit))))
