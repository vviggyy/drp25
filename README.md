# drp25
repo for Directed Reading Program SP25

#### questions...

- [ ] does adding more participants increase regression $R^2$? (0.45 -> ??)
- [ ] can we do a dimensionality reduction with a vector including macros + spike? are there any clusters --> participants are sufficiently different? 
- [ ] run analysis without fiber

#### obs...
- In order of contribution for libre in regression by coefficients: fiber (-3.13), carbs (1.01), protein (0.41). fats (0.14) and overall calories (-0.087) aren't really correlated. --> makes a lot of sense actually. doesn't say anything about contribution
- About same $R^2$ for dexcom, similar coefficient magnitudes and same signs
- Adding all participants to the same df reduced correlation? --> seems like a personalized approach is needed...

### Todo
- [ ] take care of double peaks? (prominences of 10 seem good for selecting the major peaks after meals)
- [ ] use get prominences to get left heights
- [ ] explain why there are so many different meals with 50-60 carbs. 
- [ ] improve predictive accuracy

### presentation outline
1. intro slide 
2. agenda
3. introduction about glucose monitors (personal story)
4. introductory analysis (explaining dataset, visualization of signal, healthy ranges, plotting meals, detecting peaks, explaining peaks [double peak phenomenon] )
5. first task (can we use meal macros to predict height of glucose spike)
6. user app idea 
7. different architectures of 
8. thanks to drp program (special thanks to drew and josiah)