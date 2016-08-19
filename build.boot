#!/usr/bin/env boot

(set-env! :project 'neuralnet
          :version "0.5.0"
          :source-paths #{"src"}
          ;:resource-paths #{"resources"}
          :dependencies '[[org.clojure/clojure "1.8.0"]
                          [org.clojure/tools.namespace "0.2.11"]
                          [net.mikera/vectorz-clj "0.44.1"]
                          [org.clojure/math.numeric-tower "0.0.4"]])

(use 'neuralnet.core)

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
