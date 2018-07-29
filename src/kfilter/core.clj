(ns kfilter.core
  (:require [incanter [core :as i] [charts :as c] [latex :as ltx]]  
            [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix [random :as r]]))

;; Kalman filter example demo in Python (ported to Clojure)

;; A Python implementation of the example given in pages 11-15 of "An
;; Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
;; University of North Carolina at Chapel Hill, Department of Computer
;; Science, TR 95-041,
;; https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

;; by Andrew D. Straw

;;import numpy as np
;;import matplotlib.pyplot as plt

;;plt.rcParams['figure.figsize'] = (10, 8)


;;x = -0.37727 ;; truth value (typo in example at top of p. 13 calls this z)
(def x -0.37727)
;;Q = 1e-5 ;; process variance
(def Q 1e-5)
;;R = 0.1**2 ;; estimate of measurement variance, change to see effect
(def R (Math/pow 0.1 2))

(defn init [& {:keys [guess n-iter] :or {guess 0.0 n-iter 50}}]
;; allocate space for arrays
  (let [;; intial parameters
        ;;n_iter = 50        
        ;;sz = (n_iter,) ;; size of array
        sz [n-iter]
        ;;z = np.random.normal(x,0.1,size=sz) ;; observations (normal about x, sigma=0.1)
        z (incanter.stats/sample-normal n-iter :mean x :sd 0.1)
        ;;xhat=np.zeros(sz)      ;; a posteri estimate of x
        ;;xhat[0] = guess
        
        xhat (-> (m/zero-array sz)             
                 (mp/set-1d  0 guess))
        ;;P=np.zeros(sz)         ;; a posteri error estimate
        ;;P[0] = 1.0        
        P (-> (m/zero-array sz)
              (mp/set-1d  0 1.0))
        ;;xhatminus=np.zeros(sz) ;; a priori estimate of x
        xhatminus (m/zero-array sz)
        ;;Pminus=np.zeros(sz)    ;; a priori error estimate
        Pminus (m/zero-array sz)
        ;;K=np.zeros(sz)         ;; gain or blending factor
        K (m/zero-array sz)]
    {:z z
     :xhat xhat
     :P P
     :xhatminus xhatminus
     :Pminus Pminus
     :K K
     :n-iter n-iter}))

;; for k in range(1,n_iter):
;;     ;; time update
;;     xhatminus[k] = xhat[k-1]
;;     Pminus[k] = P[k-1]+Q

;;     ;; measurement update
;;     K[k] = Pminus[k]/( Pminus[k]+R )
;;     xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
;;     P[k] = (1-K[k])*Pminus[k]

(defn experiment [ {:keys [xhat P xhatminus Pminus K n-iter z] :as input}]
  ;;standard prefix way...
  (doseq [k (range 1 n-iter)]
    (do ;; time update
      (mp/set-1d! xhatminus k  (mp/get-1d xhat (dec k)))
      (mp/set-1d! Pminus k     (+ (mp/get-1d P (dec k)) Q))
      ;; measurement update
      (mp/set-1d! K k (/ (mp/get-1d Pminus k)
                         (+ (mp/get-1d Pminus k) R)))
      ;;the mult in here may be off.    
      (mp/set-1d! xhat k (+ (* (mp/get-1d K k)
                               (- (mp/get-1d z k)
                                  (mp/get-1d xhatminus k)))
                            (mp/get-1d xhatminus k)))
      
      (mp/set-1d! P k (* (- 1 (mp/get-1d K k))
                         (mp/get-1d Pminus k)))))
  input)


;;restated to use simpler accessors, infix notation,
;;and the beginnings of functional decomposition:

;;aliases
;;shorter indexing, assignment
(def ix  mp/get-1d)
(def  <- mp/set-1d!)

;; for k in range(1,n_iter):
;;     ;; time update
;;     xhatminus[k] = xhat[k-1]
;;     Pminus[k] = P[k-1]+Q

;;     ;; measurement update
;;     K[k] = Pminus[k]/( Pminus[k]+R )
;;     xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
;;     P[k] = (1-K[k])*Pminus[k]

;;we can use incanter's infix macro, $= to defnote infix expressions.
(defn experiment-step [k {:keys [xhat P xhatminus Pminus K n-iter z] :as input}]
  (let [_ (<- xhatminus k  (i/$=  (ix xhat (k - 1))))
        _ (<- Pminus    k  (i/$=  (ix P (k - 1)) + Q))
        _ (<- K k          (i/$=  (ix Pminus k) / ((ix Pminus k) + R)))
        _ (<- xhat k       (i/$=  (ix xhatminus k) + (ix K k) * ((ix z k) - (ix xhatminus k))))
        _ (<- P k          (i/$=  (1 - (ix K k)) * (ix Pminus k)))]
    input))

(defn experiment-infix [{:keys [n-iter] :as input}]
  (doseq [k (range 1 n-iter)]
    (experiment-step k input))
  input)

;;we have additional options to pretty up the math, to include
;;making the arrays act like functions, giving a closer
;;look to the python entry access syntax.  Macros can substanially
;;aid the implementation too, if we're focusing on a bunch of
;;in-place mutation updates to bindings like the python baseline.
;;Those are left as an exercise for the reader for now.

(defn estimate-plot [{:keys [z xhat]}]
  (let [n  (mp/dimension-count z 0)
        xs (range n)]
    (-> (c/scatter-plot (range (mp/dimension-count z 0)) z
                        :x-label "iteration"
                        :y-label "voltage"
                        :title   "Estimate vs. Iteration Step"
                        :series-label "noisy measurements"
                        :legend true)
        (c/add-lines xs xhat :series-label "a posteri estimate")
        (c/add-lines xs (repeat n x) :series-label "truth value"))))

;; plt.figure()
;; plt.plot(z,'k+',label='noisy measurements')
;; plt.plot(xhat,'b-',label='a posteri estimate')
;; plt.axhline(x,color='g',label='truth value')
;; plt.legend()
;; plt.title('Estimate vs. iteration step', fontweight='bold')
;; plt.xlabel('Iteration')
;; plt.ylabel('Voltage')

(declare axis-range) ;;ephemeral hack
;; plt.figure()
;; valid_iter = range(1,n_iter) ;; Pminus not valid at step 0
;; plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
;; plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
;; plt.xlabel('Iteration')
;; plt.ylabel('$(Voltage)^2$')
;; plt.setp(plt.gca(),'ylim',[0,.01])
;; plt.show()
(defn estimated-apriori-plot
  [{:keys [Pminus] :as res}]
  (let [n  (mp/dimension-count Pminus 0)
        xs (range 1 n)]
    (-> (c/xy-plot xs
                   (map #(mp/get-1d Pminus %) xs) 
                        :x-label "iteration"
                        :y-label "Voltage^2"
                        ;;:title    #_"Estimate vs. Iteration Step"
                        
                        )
        (ltx/add-latex-subtitle "Estimated  \\it{\\mathbf{\\ a \\ priori}}  
                                 \\ error \\ vs. \\ iteration \\ step" :border [10 10 10 10])  
        (axis-range :y 0.0 0.1))))

(let [res (experiment-infix (init))]
  (i/view (estimate-plot res))
  (i/view (estimated-apriori-plot res)))

           

;;util/not important...
;;minor hack that will be incanter 1.9.4
(defn axis-range [chart axis min max]
  (let [plot (.getPlot chart)
        ax (case axis
             (:x :x-axis) (.getDomainAxis plot)
             (:y :y-axis) (.getRangeAxis   plot)
             (throw (ex-info "invalid axis!"
                             {:axis axis :expected #{:x :x-axis :y :y-axis}})))
        _ (.setRange ax min max)]
    chart))




