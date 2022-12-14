import cmath
import math
import numpy as np
from tqdm import tqdm # Progress bar
from mpmath import mp
import pandas as pd
import sys
import matplotlib.pyplot as plt



# гравитационная постоянная
G = 1
M_Sun = 1.98840987e+30
M_Jup = 1.8981246e+27


def vector_by_N(vec, N):
    """
    функция разбивает вектор vec для N, представленный как сплошной массив,
    на вектор размера (N, 3) - для каждого тела отдельный вектор
    """
    return mp.matrix(np.reshape(vec, (N, 3)))

def vector_to_N(vec):
    """
    функция обратная к vector_by_N
    """
    N = len(vec)
    return mp.matrix(np.array(vec.tolist()).reshape(N*3))


def T(p, m):
    """
    Кинетическая энергия
    :param p: вектор импульсов
    :param m: вектор масс
    :return: значение кинетической энергии
    """
    N = len(m)
    p_by_n = vector_by_N(p, N)
    T = 0
    for i in range(N):
        T += mp.norm(p_by_n[i, :]) ** mp.mpf('2') / (mp.mpf('2')*m[i])
    return T


def U(q, m):
    """
    Потенциальная энергия
    :param q: вектор в конфигурационном пространстве
    :param m: вектор масс
    :return: Значение потенциальной энергии
    """
    N = len(m)
    q_by_n = vector_by_N(p, N)
    U = 0
    for i in range(N-1):
        for j in range(i+1, N):
            U += G*m[i]*m[j] / mp.norm(q_by_n[i,:] - q_by_n[j,:])
    return U


def dT(p, m):
    """
    Производная кинетической энергии по импульсу
    :param p: вектор импульса
    :param m: вектор масс
    :return: Вектор (dT/dp_i), i = 1,...,n, где n-размерность конфигурационного пространства
    """
    N = len(m)
    p_by_n = vector_by_N(p, N)
    dT = []
    for i in range(N):
        one_dT = p_by_n[i, :] / m[i]
        dT.append(one_dT)

    dT = mp.matrix(np.array(dT))
    dT = vector_to_N(dT)
    return dT


def dU(q, m):
    """
    Производная потенцильной энергии по импульсу
    :param q: вектор координат
    :param m: вектор масс
    :return: Вектор (dU/dq_i), i = 1,...,n, где n-размерность конфигурационного пространства
    """
    N = len(m)
    q_by_n = vector_by_N(q, N)
    dU = []
    for i in range(N):
        dUi = mp.matrix([['0', '0', '0']])
        for j in range(N):
            if i == j:
                continue
            dUi += G*m[i]*m[j] * (q_by_n[i,:] - q_by_n[j,:]) / mp.norm((q_by_n[i,:] - q_by_n[j,:]))**mp.mpf('3')
        dU.append(dUi)

    dU = mp.matrix(np.array(dU))
    dU = vector_to_N(dU)
    return dU



def stormlet_verlet_explicit(q, p, m, step, n=1000, to_file=False, filename=None):
    """
    Метод Штормер-Верле, симплектический второго порядка
    :param q: вектор начальных данных по координатам
    :param p: вектор начальных данных по импульсу
    :param step: размер шага интегрирования по времени
    :param n: количество шагов для интегрирования
    :param to_file: True, если надо записывать в файл
    :param filename: имя файла для вывода результата
    :return: значения координаты и скорости на последнем шаге
    """
    if to_file:
        assert filename is not None, 'Необходимо указать имя файла'
        f = open(filename, 'w')
        original_stdout = sys.stdout
        sys.stdout = f
    for i in tqdm(range(n)):
        p_half2 = p - step / mp.mpf('2') * dU(q, m)
        q = q + step * dT(p_half2, m)
        p_n1 = p_half2 - step / mp.mpf('2') * dU(q, m)
        p = p_n1

        if to_file:
            print(i * step, ' '.join(map(lambda x: str(x[0]), q.tolist())),
                  ' '.join(map(lambda x: str(x[0]), p.tolist())))
    if to_file:
        f.close()
        sys.stdout = original_stdout
    q_3d = mp.matrix([q[0], q[1], q[2]])
    p_3d = mp.matrix([p[0], p[1], p[2]])

    return q_3d, p_3d


def to_alenas_format(file_name):
    data_frame = pd.read_csv(file_name, delimiter=' ', header=None)
    data_frame.to_csv('t.txt', header=False, columns=[0], index=False)
    j_body = 1
    for i in range(1, int((data_frame.shape[1] - 1) / 2), 3):
        data_frame.to_csv(f'XYZ-{j_body}.txt', sep=' ', index=False, header=False, columns=[i, i+1, i+2])
        j_body += 1
    j_body = 1
    for k in range(int((data_frame.shape[1] - 1) / 2 + 1), data_frame.shape[1]-1, 3):
        data_frame.to_csv(f'VxVyVz-{j_body}.txt', sep=' ', index=False, header=False, columns=[k, k + 1, k + 2])
        j_body += 1



def get_init(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = list(map(lambda x: x.replace('\n', ''), lines))

    assert len(lines) % 7 == 0
    bodies = np.array(list(zip(*[iter(lines)]*7)))

    m, q, v = bodies[:, 0], bodies[:, 1:4], bodies[:, 4:7]
    N = len(m)
    m = mp.matrix(m)
    q = mp.matrix(q)
    v = mp.matrix(v)
    p = []
    for i in range(N):
        p.append(v[i, :]*m[i])
    p = mp.matrix(np.array(p))


    q = vector_to_N(q)
    p = vector_to_N(p)

    return q, p, m


class EightOrbit:
    def __init__(self, num_sol):
        self.num_sol = num_sol
        self.time = num_sol[0]

    def set_ksi(self):
        data_frame = self.num_sol
        alpha_1 = data_frame[1]
        alpha_2 = data_frame[2]
        beta_1 = data_frame[4]
        beta_2 = data_frame[5]
        gamma_1 = data_frame[7]
        gamma_2 = data_frame[8]
        self.ksi_1 = 0.5 * ((beta_1 - alpha_1) ** 2 + (beta_2 - alpha_2) ** 2) - 0.666666 * (
                    (gamma_1 - 0.5 * (alpha_1 + beta_1)) ** 2 +
                    (gamma_2 - 0.5 * (alpha_2 + beta_2)) ** 2)
        self.ksi_3 = 2.0 / np.sqrt(3) * ((data_frame[4] - data_frame[1]) *(data_frame[8] -
        0.5 * (data_frame[2] + data_frame[5])) - (data_frame[5] - data_frame[2]) * (
                                                           data_frame[7] - 0.5 * (data_frame[1] + data_frame[4])))
        self.ksi_2 = 2 / np.sqrt(3) * (
                    (data_frame[4] - data_frame[1]) * (data_frame[7] - 0.5 * (data_frame[1] + data_frame[4])) +
                    (data_frame[5] - data_frame[2]) * (data_frame[8] - 0.5 * (data_frame[2] + data_frame[5])))
        self.ksi_1_max = np.max(abs(self.ksi_1))
        self.ksi_2_max = np.max(abs(self.ksi_2))
        self.ksi_3_max = np.max(abs(self.ksi_3))

    @staticmethod
    def phi_spherical(x, y):
        if x > 0: return np.arctan(y / x)
        elif x < 0 and y >= 0: return np.arctan(y / x) + np.pi
        elif x < 0 and y < 0: return np.arctan(y / x) - np.pi
        elif np.isclose(x, 0) and y > 0: return np.pi / 2
        elif np.isclose(x, 0) and y < 0: return -np.pi / 2

    def set_spherical_coord(self):
        self.teta = np.arccos(self.ksi_3 / (np.sqrt(self.ksi_1 * self.ksi_1 + self.ksi_2 * self.ksi_2 + self.ksi_3 * self.ksi_3)))
        self.phi = np.arctan2(self.ksi_2, self.ksi_1)
        self.ksi_1 = 0.5 * np.sin(self.teta) * np.cos(self.phi)
        self.ksi_2 = 0.5 * np.sin(self.teta) * np.sin(self.phi)
        self.ksi_3 = 0.5 * np.cos(self.teta)

    def set_riemann_coords(self, blow=1):
        self.dzeta_real = blow * self.ksi_1 / (0.5 - self.ksi_3)
        self.dzeta_imag = blow * self.ksi_2 / (0.5 - self.ksi_3)
        self.dzeta = np.array(self.dzeta_imag * 1j + self.dzeta_real)
        self.dzeta_polar = []
        for dzeta in self.dzeta:
            self.dzeta_polar.append(cmath.polar(dzeta))

    def set_lemaitre_roots(self, to_check=False):
        self.lemaitre = []
        self.euler_dzeta = [self.dzeta[0]]
        polar = cmath.polar(self.dzeta[0])
        polar_degrees = polar[1]
        polar_degrees = math.degrees(polar_degrees)
        self.euler = []
        for dzeta in self.dzeta:
            self.lemaitre.append(np.roots([1, dzeta * np.sqrt(8), 0, np.sqrt(8), - dzeta]))
            if to_check:
                check = np.roots([1, dzeta * np.sqrt(8), 0, np.sqrt(8), -dzeta])
                for t in check:
                    print(t ** 4 + dzeta * np.sqrt(8) * t ** 3 + np.sqrt(8) * t - dzeta)
        for dzeta in self.euler_dzeta:
            self.euler.append(np.roots([1, dzeta * np.sqrt(8), 0, np.sqrt(8), - dzeta]))
            if to_check:
                check = np.roots([1, dzeta * np.sqrt(8), 0, np.sqrt(8), -dzeta])
                for t in check:
                    print(t ** 4 + dzeta * np.sqrt(8) * t ** 3 + np.sqrt(8) * t - dzeta)
        self.lemaitre = np.array(self.lemaitre).flatten()
        self.euler = np.array(self.euler).flatten()
        self.euler_real = np.real(self.euler)
        self.euler_imag = np.imag(self.euler)
        self.lemaitre_real = np.real(self.lemaitre)
        self.lemaitre_imag = np.imag(self.lemaitre)

    def set_circle(self, N=10000, radius=1):
        t = np.random.random(N) * np.pi * 2
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        self.complex_circle = np.array(x + 1j * y)

    def set_dzeta_inv(self):
        self.dzeta_inv = self.complex_circle * (np.sqrt(8) + self.complex_circle ** 3) / (
                1 - np.sqrt(8) * self.complex_circle ** 3)

    def set_riemann_inv(self):
        tmp = np.abs(self.dzeta_inv) ** 2 + 1
        self.x_inv = 2 * self.dzeta_inv.real / tmp
        self.y_inv = 2 * self.dzeta_inv.imag / tmp
        self.z_inv = (np.abs(self.dzeta_inv) ** 2 - 1) / tmp
        check_tmp = self.x_inv ** 2 + self.y_inv ** 2 + self.z_inv ** 2
        assert np.max(check_tmp) - np.min(check_tmp) < 0.05, 'It is not a sphere!'


if __name__ == '__main__':
    mp.dps = 30
    masses = mp.matrix([mp.mpf('0.333333333333333333'), mp.mpf('0.333333333333333333'), mp.mpf('0.333333333333333333')])
    x_2_dot = mp.mpf('0.749442191077792')
    y_2_dot = mp.mpf('1.1501789857502275')
    init_vel = mp.matrix([mp.mpf('-0.5') * x_2_dot, mp.mpf('-0.5') * y_2_dot, mp.mpf('0.0'),  # first
                          x_2_dot, y_2_dot,  mp.mpf('0.0'),  # second
                          mp.mpf('-0.5') * x_2_dot, mp.mpf('-0.5') * y_2_dot, mp.mpf('0.0')])  # third
    init_vel_p = init_vel * masses[0]
    x_10 = masses[0] * masses[0] * (mp.mpf('2.0') * np.sqrt(mp.mpf('2.0')) + 1) / \
           (np.sqrt(mp.mpf('2.0')) * mp.mpf('-0.5') - np.sum(np.dot(init_vel, init_vel)) / 2 / masses[0])
    x_10 = mp.mpf('5') * masses[0] * masses[0] / (1 + masses[0] * np.dot(init_vel, init_vel))
    x_10 = mp.mpf(str(x_10))
    init_pos = mp.matrix([-x_10, mp.mpf('0.0'), mp.mpf('0.0'), # first
                          mp.mpf('0.0'), mp.mpf('0.0'), mp.mpf('0.0'),  # second
                          x_10, mp.mpf('0.0'), mp.mpf('0.0')])  # third
    h = mp.mpf('1e-3')
    n = 1680
    stormlet_verlet_explicit(init_pos, init_vel_p, masses, step=h, n=n, to_file=True, filename='result.dat')
    data_frame = pd.read_csv('result.dat', delimiter=' ', header=None)
    fig, ax = plt.subplots()
    plt.plot(data_frame[1], data_frame[2])
    fig.set_figwidth(8)  # ширина и
    fig.set_figheight(8)  # высота "Figure"
    ax.set_xlabel('Ось x', fontsize=14)
    ax.set_ylabel('Ось y', fontsize=14)
    plt.xlim(-0.35, 0.35)
    plt.ylim(-0.35, 0.35)
    plt.legend()
    plt.show()
    eight = EightOrbit(data_frame)
    eight.set_ksi()

