#!/usr/bin/env boot

(set-env! :project 'neuralnet
          :version "0.5.0"
          :source-paths #{"src"}
          :resource-paths #{"resources"}
          :dependencies '[[org.clojure/clojure "1.8.0"]
                          [org.clojure/tools.namespace "0.2.11"]])

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

(defn abs [n]
  (max n (- n)))

(defn sigmoid [t]
  (/ t (+ 1 (abs t))))

(defn neural-net [num-inputs num-hidden-layers num-layer-nodes num-outputs bias]
  (->> (repeat num-layer-nodes 1.0)
       (vec)
       (repeat num-layer-nodes)
       (vec)
       (repeat (dec num-hidden-layers))
       (vec)))

(def nn
  {:num-inputs 2
   :num-hidden-layers 2
   :num-outputs 1
   :weights [;all layers
             [;layer 0
              [;weights from layer 0 output 1 to layer 1
               w011
               w012
              ]
              [;weights from layer 0 output 2 to layer 1
               w021
               w022
              ]
             ]
             [;layer 2
              [;weights from layer 1 output 1 to layer 2
               w211
               w212
              ]
              [;weights from layer 1 output 1 to layer 2
               w221
               w222
              ]
             ]
             [;output layer
              [;weights for single output neuron
               w011
               w021
              ]
             ]
            ]})

(defn sigmoid [z]
  )

(for [layer (:weights nn)]
  (for [output layer]
    (for [target output]
      ;this weight goes to target in layer from previous output
      )))

(defn percept-layer [input-vals input-weights activation-function]
  )

(defn percept [input-val input-weights activation-function])



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
