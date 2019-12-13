import numpy as np

def check_column(sudoku, n, pos,inicial):
    """
    Revisa si hay numeros repetidos en columnas. 
    """
    columna = sudoku[:,pos[1]]
    contador = 0
    for i in columna:
        if i == n:
            contador += 1
    
    if inicial:
        if contador == 1:
            return True
        else:
            return False 
    else:
        if contador >= 1:
            return False
        else:
            return True

def check_row(sudoku, n, pos, inicial):
    """
    Revisa si hay numeros repeditos en filas.
    """
    fila = sudoku[pos[0],:]
    contador = 0
    for i in fila:
        if i == n:
            contador += 1
    if inicial:
        if contador == 1:
            return True 
        else:
            return False 
    else:
        if contador >= 1:
            return False 
        else:
            return True

def check_nonet(sudoku, n, pos, inicial):
    """
    Revisar si hay valores repetidos en cada cuadrito. 
    """
    y = pos[0]
    x = pos[1]
    filas = 0
    columnas = 0
    if y > 2:
        if y > 5:
            filas = 2
        else:
            filas = 1
    else:
        filas = 0

    if x > 2:
        if x > 5:
            columnas = 2
        else:
            columnas = 1
    else:
        columnas = 0
    # a este punto ya se cual cuadrito buscar 
    nonet = sudoku[filas * 3: (filas * 3) + 3, columnas * 3: (columnas * 3) + 3]
    contador = 0
    for i in range(len(nonet)):
        for j in range(len(nonet[i])):
            if nonet[i,j] == n:
                contador += 1
    if inicial:
        if contador == 1:
            return True 
        else:
            return False
    else:
        if contador >= 1:
            return False
        else:
            return True

def next_zero(sudoku, pos):
    """
    Encuentra la posición del siguiente 0.
    """
    #en la misma fila 
    for i in range(pos[1], 9):
        if sudoku[pos[0], i] == 0:
            return ((pos[0], i))
    #paso aqui es en otra fila 
    if pos[0] + 1 < 9:
        for i in range(pos[0] + 1, len(sudoku)):
            for j in range(len(sudoku[i])):
                if(sudoku[i][j] == 0):
                    return ((i,j))
    return(-1,-1)

def validity(sudoku):
    """
    Valida el sudoku inicial.
    """
    for i in range(len(sudoku)):
        for j in range(len(sudoku[i])):
            if sudoku[i,j] != 0:
                value = sudoku[i,j]
                check1 = check_column(sudoku, value, (i,j), True)
                check2 = check_row(sudoku, value, (i,j), True)
                check3 = check_nonet(sudoku, value, (i,j), True)
                if not(check1 and check2 and check3):
                    return False
    return True

def solve(sudoku, position, zeros):
    """
    Resolución de Sudoku por medio de Backtracking. 
    """
    position = position
    zeros = zeros
    while position != (-1,-1):
        value = sudoku[position[0], position[1]]
        value += 1 
        if value > 9:
            #continuar backtracking 
            actual = zeros.index((position[0],position[1]))
            if actual > 0:
                destino = zeros[actual - 1]
                #backtracking 
                sudoku[position[0],position[1]] = 0
                #solve(sudoku, destino, zeros)
                position = destino
        else: 
            for i in range(value, 10):
                check1 = check_column(sudoku,i,position, False)
                check2 = check_row(sudoku,i,position, False)
                check3 = check_nonet(sudoku,i,position, False)
                if check1 and check2 and check3:
                    sudoku[position[0],position[1]] = i
                    continuar = next_zero(sudoku, position)
                    #solve(sudoku, continuar,zeros)
                    position = continuar
                    break 
                else:
                    #solo hago backtracking si es nueve 
                    if i == 9:
                        #esto significa que ya se va a salir del loop y no ha tenido exito
                        actual = zeros.index((position[0],position[1]))
                        if actual > 0:
                            destino = zeros[actual - 1]
                            #backtracking 
                            sudoku[position[0],position[1]] = 0
                            #solve(sudoku, destino, zeros)
                            position = destino

def findzeros(sudoku):
    """
    Búsqueda de 0s en la matriz inicial.
    """
    ceros = []
    for i in range(len(sudoku)):
        for j in range(len(sudoku[i])):
            if sudoku[i,j] == 0:
                ceros.append((i,j))
    return ceros

def start(sudoku):
    """
    Inicialización.
    """
    sudoku = np.asarray(sudoku)
    ceros = findzeros(sudoku)

    continuar = validity(sudoku)
    if(continuar):
        comienzo = next_zero(sudoku, (0,0))
        solve(sudoku, comienzo, ceros)
        return(sudoku)
    else:
        return("Al parecer el sudoku no es valido...")
