import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import time

class QLearningHVAC:
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        
        # Acciones posibles: aumentar, disminuir o mantener temperatura
        self.actions = ['increase', 'decrease', 'maintain']
        
    def get_state(self, current_temp, target_temp, time_of_day):
        # Discretizar el estado para reducir el espacio de estados
        temp_diff = round((target_temp - current_temp) / 0.5) * 0.5
        return (temp_diff, time_of_day)
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.actions, key=lambda a: self.q_table[state][a])
    
    def calculate_reward(self, temp_diff, energy_used):
        comfort_penalty = -(abs(temp_diff) ** 2)
        energy_penalty = -energy_used
        return comfort_penalty + energy_penalty
    
    def update_q_value(self, state, action, reward, next_state):
        best_next_action = max(self.actions, key=lambda a: self.q_table[next_state][a])
        old_q = self.q_table[state][action]
        next_max_q = self.q_table[next_state][best_next_action]
        
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * next_max_q - old_q)
        self.q_table[state][action] = new_q

def simulate_temperature_change(current_temp, action, external_temp):
    if action == 'increase':
        change = 0.5
        energy_used = 1.0
    elif action == 'decrease':
        change = -0.5
        energy_used = 1.2
    else:  # maintain
        change = 0.1 * (external_temp - current_temp)
        energy_used = 0.3
    
    return current_temp + change, energy_used

def main():
    st.title("🌡️ Control HVAC Inteligente con Q-Learning")
    
    # Inicialización de variables en la sesión
    if 'agent' not in st.session_state:
        st.session_state.agent = QLearningHVAC()
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'episode' not in st.session_state:
        st.session_state.episode = 0
    
    # Configuración en la barra lateral
    st.sidebar.header("📊 Parámetros de Control")
    target_temp = st.sidebar.slider("Temperatura Objetivo (°C)", 18.0, 28.0, 22.0, 0.5)
    external_temp = st.sidebar.slider("Temperatura Externa (°C)", 15.0, 35.0, 25.0, 0.5)
    
    # Controles principales
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Ejecutar Episodio"):
            with st.spinner("Simulando episodio..."):
                current_temp = external_temp
                total_reward = 0
                temp_history = []
                energy_history = []
                actions_history = []
                
                # Simular 24 horas
                for hour in range(24):
                    state = st.session_state.agent.get_state(current_temp, target_temp, hour)
                    action = st.session_state.agent.get_action(state)
                    
                    new_temp, energy_used = simulate_temperature_change(current_temp, action, external_temp)
                    reward = st.session_state.agent.calculate_reward(target_temp - new_temp, energy_used)
                    
                    next_state = st.session_state.agent.get_state(new_temp, target_temp, (hour + 1) % 24)
                    st.session_state.agent.update_q_value(state, action, reward, next_state)
                    
                    temp_history.append(new_temp)
                    energy_history.append(energy_used)
                    actions_history.append(action)
                    total_reward += reward
                    current_temp = new_temp
                
                st.session_state.history.append({
                    'episode': st.session_state.episode,
                    'temperatures': temp_history,
                    'energy': energy_history,
                    'actions': actions_history,
                    'total_reward': total_reward
                })
                st.session_state.episode += 1
    
    with col2:
        if st.button("🔄 Reiniciar Simulación"):
            st.session_state.agent = QLearningHVAC()
            st.session_state.history = []
            st.session_state.episode = 0
            st.experimental_rerun()
    
    # Mostrar resultados
    if st.session_state.history:
        latest = st.session_state.history[-1]
        
        # Crear DataFrame para las gráficas
        df_temp = pd.DataFrame({
            'Hora': range(24),
            'Temperatura Actual': latest['temperatures'],
            'Temperatura Objetivo': [target_temp] * 24
        })
        
        # Gráfico de temperatura
        st.subheader("📈 Control de Temperatura")
        st.line_chart(df_temp.set_index('Hora'))
        
        # Gráfico de consumo energético
        st.subheader("⚡ Consumo Energético por Hora")
        df_energy = pd.DataFrame({
            'Hora': range(24),
            'Consumo': latest['energy']
        })
        st.bar_chart(df_energy.set_index('Hora'))
        
        # Historial de acciones
        st.subheader("🎮 Acciones Tomadas")
        df_actions = pd.DataFrame({
            'Hora': range(24),
            'Acción': latest['actions']
        })
        st.dataframe(df_actions, use_container_width=True)
        
        # Historial de recompensas si hay más de un episodio
        if len(st.session_state.history) > 1:
            st.subheader("🎯 Evolución del Aprendizaje")
            df_rewards = pd.DataFrame({
                'Episodio': range(len(st.session_state.history)),
                'Recompensa': [h['total_reward'] for h in st.session_state.history]
            })
            st.line_chart(df_rewards.set_index('Episodio'))
        
        # Métricas clave
        st.subheader("📊 Métricas del Último Episodio")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Temperatura Promedio",
                value=f"{np.mean(latest['temperatures']):.1f}°C",
                delta=f"{np.mean(latest['temperatures']) - target_temp:.1f}°C"
            )
        with col2:
            st.metric(
                label="Consumo Total",
                value=f"{sum(latest['energy']):.1f} kWh"
            )
        with col3:
            st.metric(
                label="Recompensa Total",
                value=f"{latest['total_reward']:.1f}",
                delta=f"{latest['total_reward'] - st.session_state.history[-2]['total_reward']:.1f}" if len(st.session_state.history) > 1 else None
            )
        
        # Estadísticas adicionales
        st.subheader("📝 Resumen de Rendimiento")
        tiempo_objetivo = sum(1 for t in latest['temperatures'] if abs(t - target_temp) <= 0.5)
        st.info(f"Tiempo en rango objetivo: {(tiempo_objetivo/24)*100:.1f}% del día")
        
        # Distribución de acciones
        st.subheader("🎯 Distribución de Acciones")
        action_counts = pd.DataFrame(latest['actions']).value_counts()
        st.bar_chart(action_counts)

if __name__ == "__main__":
    main()
