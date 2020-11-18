import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter


class LoRSI():
    
    def __init__(self, data_path, event_col, time_col, group_col):
        self.data = pd.read_csv(data_path)
        self.event_col = event_col
        self.time_col = time_col
        self.group_col = group_col
        self.data_filter = self.data[self.group_col] == self.data[self.group_col].unique()[0]
    
    def plot_original_KM(self):
        time = self.data[self.time_col]
        # convert to years
        if max(time) > 20:
            time = time / 365
        event = self.data[self.event_col]
        first_group = self.data[self.data_filter]
        second_group = self.data[~self.data_filter]
        kmf = KaplanMeierFitter()
        kmf.fit(time[self.data_filter], event[self.data_filter], 
                label='{} (n = {})'.format(self.data[self.group_col].unique()[0],
                                           self.data[self.data_filter].shape[0]))
        ax = kmf.plot()
        kmf.fit(time[~self.data_filter], event[~self.data_filter], 
                label='{} (n = {})'.format(self.data[self.group_col].unique()[1],
                                           self.data[~self.data_filter].shape[0]))
        kmf.plot()
        ax.set_xlim(0,10)
        ax.set_xlabel('years')
        results = logrank_test(time[self.data_filter], time[~self.data_filter], 
                               event[self.data_filter], event[~self.data_filter])
        # placeholder for the p-value
        ax.plot(0, 0, c='w', label='p-value={:.4f}'.format(results.p_value))
        ax.legend(loc='lower left')
        
    def update_data_filter(self, better_survival_group):
        self.data_filter = self.data[self.group_col] == better_survival_group
        
    def calc_interval(self, delta, delta_model):
        delta_number = int(delta * self.data.shape[0])
        if delta_model == 'RIGHT':
            max_pvalue_ommits = delta_number
            min_pvalue_ommits = 0
        elif delta_model == 'LEFT':
            max_pvalue_ommits = 0
            min_pvalue_ommits = delta_number
        else:
            delta_number = int(delta_number / 2)
            max_pvalue_ommits = delta_number
            min_pvalue_ommits = delta_number
        max_pvalue = self._get_max_pvalue(max_pvalue_ommits)
        min_pvalue = self._get_min_pvalue(min_pvalue_ommits)
        print('MIN p-value: {}'.format(min_pvalue))
        print('MAX p-value: {}'.format(max_pvalue))
    
    def _change_filter(self, changed_indexes):
        new_filter = self.data_filter.copy()
        new_filter[changed_indexes] = ~new_filter[changed_indexes]
        return new_filter
    
    def _get_max_pvalue(self, num_of_ommits):
        time = self.data[self.time_col]
        event = self.data[self.event_col]
        group_data = self.data[self.data_filter]
        non_group_data = self.data[~self.data_filter]
        event_group_data = group_data[group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=False)
        event_non_group_data = non_group_data[non_group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=True)
        censorship_non_group_data = non_group_data[non_group_data[self.event_col] == 0].sort_values(by=self.time_col, ascending=False)
        event_group_data_index = event_group_data.index
        event_non_group_data_index = event_non_group_data.index
        censorship_non_group_data_index = censorship_non_group_data.index
        current_group_event = 0
        current_non_group_event = 0
        current_non_group_censorship = 0
        max_pvalue = 0
        for i in range(num_of_ommits + 1):
            changed_filter = self._change_filter(event_group_data_index[current_group_event])
            res_group_event = logrank_test(time[changed_filter], time[~changed_filter], 
                                           event[changed_filter], event[~changed_filter])
            changed_filter = self._change_filter(event_non_group_data_index[current_non_group_event])
            res_non_group_event = logrank_test(time[changed_filter], time[~changed_filter],
                                               event[changed_filter], event[~changed_filter])
            changed_filter = self._change_filter(censorship_non_group_data_index[current_non_group_censorship])
            res_non_group_censorship = logrank_test(time[changed_filter], time[~changed_filter],
                                                    event[changed_filter], event[~changed_filter])

            results = np.array([res_group_event.p_value, res_non_group_event.p_value, 
                                res_non_group_censorship.p_value])
            max_index = np.argmax(results)
            max_pvalue = results[max_index]
            if max_index == 0:
                current_group_event += 1
            elif max_index == 1:
                current_non_group_event += 1
            else:
                current_non_group_censorship += 1
        return max_pvalue

    def _get_min_pvalue(self, num_of_ommits):
        time = self.data[self.time_col]
        event = self.data[self.event_col]
        group_data = self.data[self.data_filter]
        non_group_data = self.data[~self.data_filter]
        event_group_data = group_data[group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=True)
        event_non_group_data = non_group_data[non_group_data[self.event_col] == 1].sort_values(by=self.time_col, ascending=False)
        censorship_group_data = group_data[group_data[self.event_col] == 0].sort_values(by=self.time_col, ascending=False)
        event_group_data_index = event_group_data.index
        event_non_group_data_index = event_non_group_data.index
        censorship_group_data_index = censorship_group_data.index
        current_group_event = 0
        current_non_group_event = 0
        current_group_censorship = 0
        min_pvalue = 0
        for i in range(num_of_ommits + 1):
            changed_filter = self._change_filter(event_group_data_index[current_group_event])
            res_group_event = logrank_test(time[changed_filter], time[~changed_filter], 
                                           event[changed_filter], event[~changed_filter])
            changed_filter = self._change_filter(event_non_group_data_index[current_non_group_event])
            res_non_group_event = logrank_test(time[changed_filter], time[~changed_filter],
                                               event[changed_filter], event[~changed_filter])
            changed_filter = self._change_filter(censorship_group_data_index[current_group_censorship])
            res_group_censorship = logrank_test(time[changed_filter], time[~changed_filter],
                                                event[changed_filter], event[~changed_filter])

            results = np.array([res_group_event.p_value, res_non_group_event.p_value, 
                                res_group_censorship.p_value])
            min_index = np.argmin(results)
            min_pvalue = results[min_index]
            if min_index == 0:
                current_group_event += 1
            elif min_index == 1:
                current_non_group_event += 1
            else:
                current_group_censorship += 1
        return min_pvalue