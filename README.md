# Loan-Application-Predictor
\documentclass[11pt]{article}
\usepackage{amsfonts,amsmath,amssymb}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{geometry}
\geometry{
    paperwidth=8.5in,  % Width of the page
    paperheight=11in,  % Height of the page
    top=1in,           % Top margin
    bottom=0.5in,        % Bottom margin
    left=0.5in,          % Left margin
    right=0.5in          % Right margin
}
\documentclass{article}
\usepackage{array}
\usepackage{booktabs}
\usepackage{tcolorbox}

\begin{document}
\begin{center}
    \begin{Large}
        \textbf{MTH686 : Non-Linear Regression}\\
        \textbff{Project Report}

        
    \end{Large}
     
\end{center}
\vspace{2mm}
\textbf{Name:- Daphal Sanket Anil \\Roll No:- 210300} \\ \hline
\vspace{2mm}

 
\textbf{This project aims to thoroughly analyze a dataset in the form of $(t, y(t))$, where $t$ is generated as a sequence of independent and identically distributed (i.i.d.) normal random variables with mean zero and variance $\sigma^2$. The observations $y(t)$ are assumed to originate from one of three parametric models. Our goal is to explore each model's suitability and accuracy in capturing the underlying patterns of the data.}\\


\noindent\textbf{We have given three models :-}

\begin{itemize}
    \item \textbf{Model - 1 :}  $$y(t) = \alpha_0 + \alpha_1 e^{\beta_1 t} + \alpha_2 e^{\beta_2 t} + \epsilon(t)$$

    \item \textbf{Model - 2 :} $$y(t) = \frac{\alpha_0 + \alpha_1 t}{\beta_0 + \beta_1 t} + \epsilon(t)
$$
    \item \textbf{Model - 3 :} $$y(t) = \beta_0 + \beta_1 t + \beta_2 t^2 + \beta_3 t^3 + \beta_4 t^4 + \epsilon(t)
$$
\end{itemize}
\{\epsilon(t) \} \text{ is a sequence of i.i.d. normal random variables with mean } 0 \text{ and variance } \sigma^2.


\\
\\
\vspace{4mm}

 

 

\noindent\underline{\textbf{Model 1 Least Square Estimator}}:-\\ \\ 
 $$y(t) = \alpha_0 + \alpha_1 e^{\beta_1 t} + \alpha_2 e^{\beta_2 t} + \epsilon(t)$$
{Parameters:}

\begin{itemize}
    \item $\alpha_0$, $\alpha_1$, $\alpha_2$, $\beta_1$, $\beta_2$
\end{itemize} 

if we assume \( \beta_1 \) and \( \beta_2 \), then we can estimate the $\alpha$ by simple Linear Regression model
\\

$$Y = X(\beta)(\alpha) +\epsilon$$
Where $$\alpha = [\alpha_0,\alpha_1,\alpha_2]^T$$
\[
X = 
\begin{bmatrix}
    1 & e^{\beta_1 t_1} & e^{\beta_2 t_1} \\
    1 & e^{\beta_1 t_2} & e^{\beta_2 t_2} \\
    \vdots & \vdots & \vdots \\
    1 & e^{\beta_1 t_n} & e^{\beta_2 t_n} \\
\end{bmatrix}
\]

\\

Define :
$$Q_{LSE}(\alpha,\beta) = ((Y - X(\beta)(\alpha))^T(Y - X(\beta)(\alpha))$$
\\

\noindent we need to minimize the $Q_{LSE}(\alpha,\beta)$ w.r.t \alpha , \beta . 
\\
\noindent Estimate,\( \alpha \) Values :
Using   \( \hat{\alpha} = (X^T X)^{-1} X^T Y \), solve for the \(\alpha\) coefficients.
$$
\hat{\alpha}(\beta)
= (X(\beta)^T X(\beta)^{-1} X(\beta)^T Y
\]
$$

After obtaining the values for $\alpha_0$, $\alpha_1$, $\alpha_2$ given initial guesses for $\beta_1$ and $\beta_2$ the model reduces to having only $\beta_1$ and $\beta_2$ as unknown parameters. At this stage, we can utilize a numerical optimization method to determine the optimal values for  $\beta_1$ and $\beta_2$.
\\
$$\hat{\beta}_{\text{LSE}} = \arg \min Q_{\text{LSE}}(\beta)
$$
\\
To find the values of \( \beta_1 \) and \( \beta_2 \) that minimize the sum of squared residuals (our objective function \( Q \)), we used Python's built-in function \texttt{minimize} from the \texttt{scipy.optimize} library. This function performs iterative optimization to identify the values of \( \beta_1 \) and \( \beta_2 \) that provide the best fit by minimizing \( Q \).
\\
\\
\textbf{\underline{Calculate the Estimated \( Y \)-values}:}

 
\[
Y_{\text{est}} = X_{\beta} \cdot \hat{\alpha}
\]
 

After performing the least squares estimation and optimization for Model 1, the following values were obtained for the parameters:

\begin{table}[h!]
\centering
\begin{tabular}{|c|c|}
\hline
\textbf{Parameter} & \textbf{Value} \\
\hline
\( \alpha_0 \) & 3.498 \\
\( \alpha_1 \) &  4.688 \\
\( \alpha_2 \) & -0.132 \\
\( \beta_1 \) & 1.521\\
\( \beta_2 \) & -3.734\\
\hline
\end{tabular}
\end{table}



 

\noindent\textbf{\underline{Model 2 Least Square Estimator} :- }\\ \\ 
Equation: $$y(t) = \frac{\alpha_0 + \alpha_1 t}{\beta_0 + \beta_1 t} + \varepsilon(t) , 
$$
 
We will find the least square estimator of this moddel using same apporach that we used in model 1 case.\\
if we know \( \beta_0 \) and \( \beta_1 \), then we can estimate the $\alpha$ by simple Linear Regression model
\\
$$Y = X(\beta)(\alpha) +\epsilon$$
\[
X(\beta)  = 
\begin{bmatrix}
\frac{1}{\beta_0 + \beta_1} & \frac{1}{\beta_0 + \beta_1} \\
\frac{1}{\beta_0 + 2\beta_1} & \frac{2}{\beta_0 + 2\beta_1} \\
\vdots & \vdots \\
\frac{1}{\beta_0 + n\beta_1} & \frac{n}{\beta_0 + n\beta_1} \\
\end{bmatrix}
\]



\noindent Define :
$$Q_{LSE}(\alpha,\beta) = (Y - X(\beta)(\alpha)^T(Y - X(\beta)(\alpha))$$
we need to minimize the $Q_{LSE}(\alpha,\beta)$ w.r.t \alpha , \beta . 

\\
\newpage
\noindent Estimate \( \alpha \) Values :
Using   \( \hat{\alpha} = (X^T X)^{-1} X^T Y \), solve for the \(\alpha\) coefficients.
$$
\hat{\alpha}(\beta)
= (X(\beta)^T X(\beta)^{-1} X(\beta)^T Y
\]
$$

After obtaining the values for $\alpha_0$, $\alpha_1$, $\alpha_2$ given initial guesses for $\beta_1$ and $\beta_2$ the model reduces to having only $\beta_1$ and $\beta_2$ as unknown parameters. At this stage, we can utilize a numerical optimization method to determine the optimal values for  $\beta_1$ and $\beta_2$.
\\
$$Q_{\text{LSE}}(\beta) = (Y - X(\beta) \hat{\alpha}(\beta))^T (Y - X(\beta) \hat{\alpha}(\beta))
$$
\vspace{1mm}
$$\hat{\beta}_{\text{LSE}} = \arg \min Q_{\text{LSE}}(\beta)
$$

 To find the values of \( \beta_1 \) and \( \beta_2 \) that minimize the sum of squared residuals (our objective function \( Q \)), we used Python's built-in function \texttt{minimize} from the \texttt{scipy.optimize} library. This function performs iterative optimization to identify the values of \( \beta_1 \) and \( \beta_2 \) that provide the best fit by minimizing \( Q \).
\\
 
\noindent\textbf{\underline{Calculate the Estimated \( Y \)-values}:}

 
\[
Y_{\text{est}} = X_{\beta} \cdot \hat{\alpha}
\]
 

After performing the least squares estimation and optimization for Model 1, the following values were obtained for the parameters:

\begin{table}[h!]
\centering
\begin{tabular}{|c|c|}
\hline
\textbf{Parameter} & \textbf{Value} \\
\hline
\( \alpha_0 \) &  6.307 \\
\( \alpha_1 \) & 3.364 \\
\( \beta_0 \) & 0.793 \\
\( \beta_1 \) & -0.408

 \\
\hline
\end{tabular}
\end{table}

 
 
\noindent\textbf{\underline{Model 3 Least Square Estimator} :- }\\ \\ 
Equation: $$y(t) = \beta_0 + \beta_1 t + \beta_2 t^2 + \beta_3 t^3 + \beta_4 t^4 + \epsilon(t)
$$
{Parameters:}

\begin{itemize}
    \item \beta_0$, $\beta_1$, $\beta_2$, $\beta_3$,$\beta_4$
\end{itemize} 
take $\beta = [\beta_0, \dots, \beta_4]^T$ , take other things as same as in model 1 and 2,
we can write mode 3 as :
 $$Y = X\beta + \epsilon(t)$$
 \\
$$X =  
\begin{bmatrix}
1 & 1 & 1 & 1 & 1 \\
1 & 2 & 2^2 & 2^3 & 2^4 \\
1 & 3 & 3^2 & 3^3 & 3^4 \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
1 & n & n^2 & n^3 & n^4 \\
\end{bmatrix}$$
\\
it is simple linear regression model.
we know LSE of this is :
$$\( \hat{\beta} = (X^T X)^{-1} X^T Y \)$$
 


\noindent\textbf{\underline{Calculate the Estimated \( Y \)-values}:}

 
\[
Y_{\text{est}} = X_{\beta} \cdot \hat{\alpha}
\]
 

After performing the least squares estimation and optimization for Model 3, the following values were obtained for the parameters:

\begin{table}[h!]
\centering
\begin{tabular}{|c|c|}
\hline
\textbf{Parameter} & \textbf{Value} \\
\hline
\( \beta_0 \) & 8.054 \\
\( \beta_1 \) & 7.600 \\
\( \beta_2 \) & 4.893 \\
\( \beta_3 \) & 2.526 \\
\( \beta_4 \) & 1.894 \\
\hline
\end{tabular}
\end{table}
\noindent\textbf{\underline{Best fitted model} :-}For the optimal model, we focus on the one that minimizes \( Q_{\text{LSE}} \), bringing it as close to zero as possible. This selection is based on comparing the Least Squares Errors (LSE) across different models.
\[
\boxed{
\begin{array}{l}
Q_{\text{LSE}}(\text{Model 1}) = 0.46667\\
Q_{\text{LSE}}(\text{Model 2}) = 0.64518\\
Q_{\text{LSE}}(\text{Model 3}) = 0.46601
\end{array}
}
\]
from above we get that \textbf{best fitted model is model - 3}
\\
\textbf{\underline{Confidence Interval for Model 3} :-}
\begin{table}[h!]
\centering
\begin{tabular}{|c|c|}
\hline
\textbf{Parameter} & \textbf{Value} \\
\hline
 
            $\beta_0$ & [7.9528, 8.1554] \\
            $\beta_1$ & [6.2354, 8.9648] \\
            $\beta_2$ & [-0.5344, 10.3210] \\
            $\beta_3$ & [-5.4986, 10.5509] \\
            $\beta_4$ & [-2.0358, 5.8239] \\
\hline
\end{tabular}
\end{table}
\\
\noindent\textbf{\underline{estimate of }\sigma^2}:-
\[
\boxed{
\begin{array}{l}
\text{Model 1: } \hat{\sigma}^2 = 0.006666 \\
\text{Model 2: } \hat{\sigma}^2 = 0.009087 \\
\text{Model 3: } \hat{\sigma}^2 = 0.006657
\end{array}
}
\]

 \noindent\textbf{\underline{Residue Plot }:- }

 \begin{figure}[ht]
    \centering
    % First figure
    \begin{minipage}{0.3\textwidth}
        \centering
        \includegraphics[width=\linewidth]{1.1.png}
        \caption{Model - 1}
    \end{minipage}
    \hfill
    % Second figure
    \begin{minipage}{0.3
\textwidth}
        \centering
        \includegraphics[width=\linewidth]{2.1.png}
        \caption{Model 2}
    \end{minipage}
    \hfill
    % Third figure
    \begin{minipage}{0.3\textwidth}
        \centering
        \includegraphics[width=\linewidth]{3.1.png}
        

        \caption{Model 3}
    \end{minipage}
\end{figure}
\newpage
 \textbf{\underline{Q-Q Plot }:- }

 \begin{figure}[ht]
    \centering
    
    % First figure
    \begin{minipage}{0.46\textwidth}
        \centering
        \includegraphics[width=\linewidth]{1.2.png}
        \vspace{-4mm}
        \caption{Model - 1}
    \end{minipage}
    \hfill
    % Second figure
    \begin{minipage}{0.46\textwidth}
        \centering
        \includegraphics[width=\linewidth]{2.2.png}
        \caption{Model 2}
    \end{minipage}
    \hfill
    % Third figure
    \begin{minipage}{0.46\textwidth}
        \centering
        \includegraphics[width=\linewidth]{3.2.png}
        \caption{Model 3}
    \end{minipage}
\end{figure}
\newpage
 \textbf{\underline{Actual vs Estimated}:- }

 \begin{figure}[ht]
    \centering
    % First figure
    \begin{minipage}{0.45\textwidth}
        \centering
        \includegraphics[width=\linewidth]{1.3.png}
        \caption{Model - 1}
    \end{minipage}
    \hfill
    % Second figure
    \begin{minipage}{0.45\textwidth}
        \centering
        \includegraphics[width=\linewidth]{2.3.png}
        \caption{Model 2}
    \end{minipage}
    \hfill
    % Third figure
    \begin{minipage}{0.45\textwidth}
        \centering
        \includegraphics[width=\linewidth]{3.3.png}

        \caption{Model 3}
    \end{minipage}
\end{figure}
\end{document}
