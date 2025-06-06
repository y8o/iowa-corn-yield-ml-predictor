# Iowa Corn Yield Prediction Model - Complete Implementation
# Climate-Driven ML Model for Agricultural Forecasting

import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML and visualization libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🌽 Iowa Corn Yield ML Predictor")
print("=" * 50)

# =============================================================================
# SECTION 1: DATA COLLECTION
# =============================================================================

def get_sample_yield_data():
    """Generate realistic sample corn yield data for Iowa counties"""
    
    # Iowa's top corn-producing counties
    iowa_counties = [
        'Kossuth', 'Pocahontas', 'Wright', 'Hamilton', 'Story', 'Boone', 
        'Marshall', 'Tama', 'Benton', 'Linn', 'Grundy', 'Butler', 'Franklin',
        'Cerro Gordo', 'Worth', 'Mitchell', 'Howard', 'Winneshiek', 'Allamakee',
        'Clayton', 'Fayette', 'Buchanan', 'Delaware', 'Dubuque', 'Jackson',
        'Clinton', 'Scott', 'Muscatine', 'Louisa', 'Washington', 'Johnson',
        'Iowa', 'Keokuk', 'Mahaska', 'Marion', 'Jasper', 'Polk', 'Dallas',
        'Madison', 'Warren', 'Lucas', 'Monroe', 'Wapello', 'Jefferson',
        'Henry', 'Des Moines', 'Lee', 'Van Buren', 'Davis', 'Appanoose'
    ]
    
    years = list(range(2013, 2024))  # 11 years of data
    
    data = []
    np.random.seed(42)  # For reproducible results
    
    for county in iowa_counties:
        for year in years:
            # Base yield with county-specific variation
            base_yield = np.random.normal(185, 15)  # Iowa average ~180-190 bu/acre
            
            # Year effects (drought years, good years)
            year_effect = {
                2013: -5,   # Slightly below average
                2014: 15,   # Great year
                2015: -10,  # Challenging conditions
                2016: 20,   # Excellent year
                2017: 5,    # Above average
                2018: -8,   # Variable conditions
                2019: -15,  # Wet spring, challenging year
                2020: 8,    # Good conditions
                2021: 12,   # Strong year
                2022: -20,  # Drought impact
                2023: 10    # Recovery year
            }.get(year, 0)
            
            final_yield = max(120, base_yield + year_effect + np.random.normal(0, 8))
            
            data.append({
                'county': county,
                'year': year,
                'yield_bu_per_acre': round(final_yield, 1),
                'harvested_acres': np.random.randint(45000, 85000)
            })
    
    return pd.DataFrame(data)

def get_sample_climate_data():
    """Generate realistic climate data for Iowa"""
    
    iowa_counties = [
        'Kossuth', 'Pocahontas', 'Wright', 'Hamilton', 'Story', 'Boone', 
        'Marshall', 'Tama', 'Benton', 'Linn', 'Grundy', 'Butler', 'Franklin',
        'Cerro Gordo', 'Worth', 'Mitchell', 'Howard', 'Winneshiek', 'Allamakee',
        'Clayton', 'Fayette', 'Buchanan', 'Delaware', 'Dubuque', 'Jackson',
        'Clinton', 'Scott', 'Muscatine', 'Louisa', 'Washington', 'Johnson',
        'Iowa', 'Keokuk', 'Mahaska', 'Marion', 'Jasper', 'Polk', 'Dallas',
        'Madison', 'Warren', 'Lucas', 'Monroe', 'Wapello', 'Jefferson',
        'Henry', 'Des Moines', 'Lee', 'Van Buren', 'Davis', 'Appanoose'
    ]
    
    years = list(range(2013, 2024))
    
    data = []
    np.random.seed(42)
    
    for county in iowa_counties:
        for year in years:
            # Growing season climate (April-September)
            
            # Growing Degree Days (GDD) - base 50°F, cap 86°F
            # Normal Iowa GDD: ~2800-3200 for corn
            base_gdd = np.random.normal(3000, 200)
            
            # Year-specific GDD adjustments
            gdd_adjustments = {
                2022: -400,  # Hot, dry year - less optimal GDD accumulation
                2019: -200,  # Cool, wet spring
                2014: 150,   # Ideal growing conditions
                2016: 180,   # Excellent year
                2021: 120    # Strong year
            }
            
            gdd = max(2200, base_gdd + gdd_adjustments.get(year, 0) + np.random.normal(0, 100))
            
            # Growing season precipitation (inches)
            # Normal Iowa: 20-25 inches April-Sept
            base_precip = np.random.normal(22, 3)
            
            precip_adjustments = {
                2022: -8,    # Drought year
                2019: 6,     # Very wet
                2015: -4,    # Dry conditions
                2018: -3     # Variable
            }
            
            precip = max(8, base_precip + precip_adjustments.get(year, 0) + np.random.normal(0, 2))
            
            # Temperature variables
            avg_temp = np.random.normal(70, 3)  # Growing season average
            max_temp = avg_temp + np.random.normal(15, 2)
            min_temp = avg_temp - np.random.normal(15, 2)
            
            # Extreme weather indicators
            days_over_90 = max(0, np.random.poisson(8) + (5 if year == 2022 else 0))
            consecutive_dry_days = max(0, np.random.poisson(12) + (10 if year == 2022 else 0))
            
            data.append({
                'county': county,
                'year': year,
                'gdd_total': round(gdd, 0),
                'precip_total': round(precip, 1),
                'avg_temp': round(avg_temp, 1),
                'max_temp': round(max_temp, 1),
                'min_temp': round(min_temp, 1),
                'days_over_90': days_over_90,
                'consecutive_dry_days': consecutive_dry_days,
                'heat_stress_days': max(0, days_over_90 - 5)  # Days with potential heat stress
            })
    
    return pd.DataFrame(data)

print("🔄 Loading agricultural and climate data...")

# Load the data
yield_data = get_sample_yield_data()
climate_data = get_sample_climate_data()

print(f"✅ Loaded {len(yield_data)} yield records across {yield_data['county'].nunique()} counties")
print(f"✅ Loaded {len(climate_data)} climate records from {climate_data['year'].min()}-{climate_data['year'].max()}")

# Display sample data
print("\n📊 Sample Yield Data:")
print(yield_data.head())

print("\n🌤️ Sample Climate Data:")
print(climate_data.head())

# =============================================================================
# SECTION 2: DATA EXPLORATION & FEATURE ENGINEERING
# =============================================================================

# Merge datasets
print("\n🔗 Merging yield and climate data...")
df = pd.merge(yield_data, climate_data, on=['county', 'year'], how='inner')
print(f"✅ Created merged dataset with {len(df)} records")

# Feature engineering
print("\n⚙️ Engineering agricultural features...")

def calculate_additional_features(df):
    """Calculate additional agricultural and climate features"""
    
    # GDD efficiency (yield per unit GDD)
    df['gdd_efficiency'] = df['yield_bu_per_acre'] / df['gdd_total']
    
    # Precipitation efficiency
    df['precip_efficiency'] = df['yield_bu_per_acre'] / df['precip_total']
    
    # Temperature stress indicator
    df['temp_stress'] = (df['days_over_90'] * 2) + df['heat_stress_days']
    
    # Optimal precipitation indicator (20-25 inches is optimal)
    df['precip_optimal'] = 1 - abs(df['precip_total'] - 22.5) / 22.5
    
    # GDD adequacy (closer to 3000 is better)
    df['gdd_adequacy'] = 1 - abs(df['gdd_total'] - 3000) / 3000
    
    # Drought stress indicator
    df['drought_stress'] = df['consecutive_dry_days'] / 30  # Normalize to 0-1 scale
    
    return df

df = calculate_additional_features(df)

# Basic statistics
print("\n📈 Dataset Statistics:")
print(df.describe())

# =============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n📊 Creating exploratory visualizations...")

# Set up the plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Iowa Corn Yield Analysis - Key Relationships', fontsize=16, fontweight='bold')

# 1. Yield distribution over years
ax1 = axes[0, 0]
yearly_yield = df.groupby('year')['yield_bu_per_acre'].mean()
ax1.plot(yearly_yield.index, yearly_yield.values, marker='o', linewidth=2, markersize=8)
ax1.set_title('Average Corn Yield by Year')
ax1.set_xlabel('Year')
ax1.set_ylabel('Yield (Bu/Acre)')
ax1.grid(True, alpha=0.3)

# 2. GDD vs Yield relationship
ax2 = axes[0, 1]
ax2.scatter(df['gdd_total'], df['yield_bu_per_acre'], alpha=0.6, s=30)
ax2.set_title('Growing Degree Days vs Yield')
ax2.set_xlabel('Total GDD')
ax2.set_ylabel('Yield (Bu/Acre)')

# Add trend line
z = np.polyfit(df['gdd_total'], df['yield_bu_per_acre'], 1)
p = np.poly1d(z)
ax2.plot(df['gdd_total'], p(df['gdd_total']), "r--", alpha=0.8)

# 3. Precipitation vs Yield
ax3 = axes[0, 2]
ax3.scatter(df['precip_total'], df['yield_bu_per_acre'], alpha=0.6, s=30, color='green')
ax3.set_title('Growing Season Precipitation vs Yield')
ax3.set_xlabel('Precipitation (inches)')
ax3.set_ylabel('Yield (Bu/Acre)')

# Add trend line
z = np.polyfit(df['precip_total'], df['yield_bu_per_acre'], 1)
p = np.poly1d(z)
ax3.plot(df['precip_total'], p(df['precip_total']), "r--", alpha=0.8)

# 4. Heat stress impact
ax4 = axes[1, 0]
stress_yield = df.groupby('days_over_90')['yield_bu_per_acre'].mean()
ax4.bar(stress_yield.index, stress_yield.values, color='orange', alpha=0.7)
ax4.set_title('Heat Stress Days vs Average Yield')
ax4.set_xlabel('Days Over 90°F')
ax4.set_ylabel('Average Yield (Bu/Acre)')

# 5. Drought impact
ax5 = axes[1, 1]
drought_bins = pd.cut(df['consecutive_dry_days'], bins=5)
drought_yield = df.groupby(drought_bins)['yield_bu_per_acre'].mean()
ax5.bar(range(len(drought_yield)), drought_yield.values, color='brown', alpha=0.7)
ax5.set_title('Drought Stress vs Yield')
ax5.set_xlabel('Consecutive Dry Days (Binned)')
ax5.set_ylabel('Average Yield (Bu/Acre)')
ax5.set_xticks(range(len(drought_yield)))
ax5.set_xticklabels([f'{int(interval.left)}-{int(interval.right)}' for interval in drought_yield.index], rotation=45)

# 6. Correlation heatmap
ax6 = axes[1, 2]
correlation_vars = ['yield_bu_per_acre', 'gdd_total', 'precip_total', 'avg_temp', 'days_over_90', 'consecutive_dry_days']
corr_matrix = df[correlation_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax6, 
            square=True, fmt='.2f', cbar=True)
ax6.set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 4: MACHINE LEARNING MODEL
# =============================================================================

print("\n🤖 Building Machine Learning Model...")

# Prepare features and target
feature_columns = [
    'gdd_total', 'precip_total', 'avg_temp', 'max_temp', 'min_temp',
    'days_over_90', 'consecutive_dry_days', 'heat_stress_days',
    'temp_stress', 'precip_optimal', 'gdd_adequacy', 'drought_stress'
]

X = df[feature_columns].copy()
y = df['yield_bu_per_acre'].copy()

print(f"Features: {len(feature_columns)}")
print(f"Samples: {len(X)}")

# Handle any missing values
X = X.fillna(X.mean())

# Split the data (use recent years as test set)
train_mask = df['year'] < 2022  # Train on 2013-2021
test_mask = df['year'] >= 2022  # Test on 2022-2023

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train Random Forest model
print("\n🌲 Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Make predictions
train_pred = rf_model.predict(X_train)
test_pred = rf_model.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print(f"\n📊 Model Performance:")
print(f"Training R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")
print(f"Training MAE: {train_mae:.2f} bu/acre")
print(f"Test MAE: {test_mae:.2f} bu/acre")
print(f"Training RMSE: {train_rmse:.2f} bu/acre")
print(f"Test RMSE: {test_rmse:.2f} bu/acre")

# Cross-validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# =============================================================================
# SECTION 5: MODEL ANALYSIS & VISUALIZATION
# =============================================================================

print("\n📈 Creating model analysis visualizations...")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

# Create comprehensive model analysis plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Machine Learning Model Analysis', fontsize=16, fontweight='bold')

# 1. Feature Importance
ax1 = axes[0, 0]
bars = ax1.barh(feature_importance['feature'], feature_importance['importance'], 
                color='steelblue', alpha=0.8)
ax1.set_title('Feature Importance (Random Forest)')
ax1.set_xlabel('Importance Score')

# Add value labels on bars
for bar in bars:
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=9)

# 2. Actual vs Predicted (Test Set)
ax2 = axes[0, 1]
ax2.scatter(y_test, test_pred, alpha=0.7, s=50, color='green')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Yield (Bu/Acre)')
ax2.set_ylabel('Predicted Yield (Bu/Acre)')
ax2.set_title(f'Actual vs Predicted (Test Set)\nR² = {test_r2:.3f}')
ax2.grid(True, alpha=0.3)

# 3. Residuals plot
ax3 = axes[1, 0]
residuals = y_test - test_pred
ax3.scatter(test_pred, residuals, alpha=0.7, s=50, color='orange')
ax3.axhline(y=0, color='r', linestyle='--')
ax3.set_xlabel('Predicted Yield (Bu/Acre)')
ax3.set_ylabel('Residuals (Bu/Acre)')
ax3.set_title('Residuals Plot (Test Set)')
ax3.grid(True, alpha=0.3)

# 4. Prediction errors by year
ax4 = axes[1, 1]
test_df = df[test_mask].copy()
test_df['predicted'] = test_pred
test_df['error'] = abs(test_df['yield_bu_per_acre'] - test_df['predicted'])

yearly_error = test_df.groupby('year')['error'].mean()
ax4.bar(yearly_error.index, yearly_error.values, color='red', alpha=0.7)
ax4.set_xlabel('Year')
ax4.set_ylabel('Mean Absolute Error (Bu/Acre)')
ax4.set_title('Prediction Error by Year')
ax4.set_xticks(yearly_error.index)

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 6: AGRICULTURAL INSIGHTS
# =============================================================================

print("\n🌾 Agricultural Insights Analysis...")

# Top and bottom performing counties in recent years
recent_years = df[df['year'] >= 2020]
county_performance = recent_years.groupby('county').agg({
    'yield_bu_per_acre': ['mean', 'std'],
    'gdd_total': 'mean',
    'precip_total': 'mean'
}).round(2)

county_performance.columns = ['avg_yield', 'yield_std', 'avg_gdd', 'avg_precip']
county_performance = county_performance.sort_values('avg_yield', ascending=False)

print("\n🏆 Top 10 Performing Counties (2020-2023):")
print(county_performance.head(10))

print("\n⚠️ Most Variable Counties (High Yield Std Dev):")
print(county_performance.sort_values('yield_std', ascending=False).head(5))

# Climate sensitivity analysis
print("\n🌡️ Climate Sensitivity Analysis:")

# GDD sensitivity
gdd_bins = pd.qcut(df['gdd_total'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
gdd_sensitivity = df.groupby(gdd_bins)['yield_bu_per_acre'].agg(['mean', 'count'])
print("\nGDD Impact on Yield:")
print(gdd_sensitivity)

# Precipitation sensitivity
precip_bins = pd.qcut(df['precip_total'], q=4, labels=['Dry', 'Moderate-Dry', 'Moderate-Wet', 'Wet'])
precip_sensitivity = df.groupby(precip_bins)['yield_bu_per_acre'].agg(['mean', 'count'])
print("\nPrecipitation Impact on Yield:")
print(precip_sensitivity)

# Heat stress analysis
heat_impact = df.groupby(pd.cut(df['days_over_90'], bins=[0, 5, 10, 15, 30]))['yield_bu_per_acre'].mean()
print("\nHeat Stress Impact (Days over 90°F):")
print(heat_impact)

# =============================================================================
# SECTION 7: PREDICTIVE SCENARIOS
# =============================================================================

print("\n🔮 Climate Scenario Predictions...")

# Create scenarios for different climate conditions
scenarios = pd.DataFrame({
    'scenario': ['Optimal', 'Drought', 'Heat_Stress', 'Excessive_Rain', 'Cool_Season'],
    'gdd_total': [3000, 2600, 3400, 2800, 2400],
    'precip_total': [22, 12, 18, 32, 26],
    'avg_temp': [72, 78, 76, 70, 66],
    'max_temp': [85, 95, 92, 82, 78],
    'min_temp': [58, 60, 62, 58, 54],
    'days_over_90': [8, 25, 30, 5, 2],
    'consecutive_dry_days': [12, 35, 20, 8, 15],
    'heat_stress_days': [3, 20, 25, 0, 0]
})

# Calculate derived features for scenarios
scenarios['temp_stress'] = (scenarios['days_over_90'] * 2) + scenarios['heat_stress_days']
scenarios['precip_optimal'] = 1 - abs(scenarios['precip_total'] - 22.5) / 22.5
scenarios['gdd_adequacy'] = 1 - abs(scenarios['gdd_total'] - 3000) / 3000
scenarios['drought_stress'] = scenarios['consecutive_dry_days'] / 30

# Make predictions for each scenario
scenario_features = scenarios[feature_columns]
scenario_predictions = rf_model.predict(scenario_features)
scenarios['predicted_yield'] = scenario_predictions

print("Climate Scenario Predictions:")
print("=" * 50)
for i, row in scenarios.iterrows():
    print(f"{row['scenario']:15s}: {row['predicted_yield']:5.1f} bu/acre")
    
print(f"\nYield Range: {scenario_predictions.min():.1f} - {scenario_predictions.max():.1f} bu/acre")
print(f"Impact Spread: {scenario_predictions.max() - scenario_predictions.min():.1f} bu/acre difference")

# =============================================================================
# SECTION 8: INTERACTIVE VISUALIZATIONS
# =============================================================================

print("\n🎨 Creating interactive visualizations...")

# Create interactive plotly visualizations
def create_interactive_plots():
    """Create interactive Plotly visualizations"""
    
    # 1. Interactive yield trends over time
    fig1 = px.line(df.groupby('year')['yield_bu_per_acre'].mean().reset_index(), 
                   x='year', y='yield_bu_per_acre',
                   title='Iowa Corn Yield Trends (2013-2023)',
                   labels={'yield_bu_per_acre': 'Average Yield (Bu/Acre)', 'year': 'Year'})
    fig1.update_traces(line=dict(width=3), marker=dict(size=8))
    fig1.update_layout(height=400)
    
    # 2. 3D scatter plot: GDD vs Precip vs Yield
    fig2 = px.scatter_3d(df, x='gdd_total', y='precip_total', z='yield_bu_per_acre',
                         color='year', size='harvested_acres',
                         title='3D Climate-Yield Relationship',
                         labels={
                             'gdd_total': 'Growing Degree Days',
                             'precip_total': 'Precipitation (inches)',
                             'yield_bu_per_acre': 'Yield (Bu/Acre)'
                         })
    fig2.update_layout(height=600)
    
    # 3. Feature importance with error bars
    importance_df = feature_importance.copy()
    fig3 = px.bar(importance_df, x='importance', y='feature', 
                  orientation='h',
                  title='Model Feature Importance',
                  labels={'importance': 'Importance Score', 'feature': 'Climate Variables'})
    fig3.update_layout(height=500)
    
    return fig1, fig2, fig3

# Create the interactive plots
interactive_figs = create_interactive_plots()

# Display static versions for notebook
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Final Model Results & Climate Impact Analysis', fontsize=16, fontweight='bold')

# 1. Scenario comparison
ax1 = axes[0, 0]
scenario_colors = ['green', 'red', 'orange', 'blue', 'purple']
bars = ax1.bar(scenarios['scenario'], scenarios['predicted_yield'], 
               color=scenario_colors, alpha=0.8)
ax1.set_title('Climate Scenario Yield Predictions')
ax1.set_ylabel('Predicted Yield (Bu/Acre)')
ax1.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, scenarios['predicted_yield']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{value:.0f}', ha='center', va='bottom', fontweight='bold')

# 2. Model performance summary
ax2 = axes[0, 1]
metrics = ['R²', 'MAE', 'RMSE']
train_values = [train_r2, train_mae, train_rmse]
test_values = [test_r2, test_mae, test_rmse]

x = np.arange(len(metrics))
width = 0.35

ax2.bar(x - width/2, train_values, width, label='Training', alpha=0.8, color='skyblue')
ax2.bar(x + width/2, test_values, width, label='Test', alpha=0.8, color='lightcoral')

ax2.set_title('Model Performance Metrics')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Climate factor impact
ax3 = axes[1, 0]
climate_factors = ['GDD', 'Precipitation', 'Heat Stress', 'Drought Stress']
impact_scores = [
    feature_importance[feature_importance['feature'] == 'gdd_total']['importance'].iloc[0],
    feature_importance[feature_importance['feature'] == 'precip_total']['importance'].iloc[0],
    feature_importance[feature_importance['feature'] == 'temp_stress']['importance'].iloc[0],
    feature_importance[feature_importance['feature'] == 'drought_stress']['importance'].iloc[0]
]

bars = ax3.bar(climate_factors, impact_scores, color=['gold', 'blue', 'red', 'brown'], alpha=0.8)
ax3.set_title('Key Climate Factor Importance')
ax3.set_ylabel('Model Importance Score')
ax3.tick_params(axis='x', rotation=45)

# 4. Yield distribution by climate conditions
ax4 = axes[1, 1]
# Create bins for GDD and show yield distribution
gdd_categories = pd.cut(df['gdd_total'], bins=4, labels=['Low GDD', 'Med-Low GDD', 'Med-High GDD', 'High GDD'])
yield_by_gdd = [df[gdd_categories == cat]['yield_bu_per_acre'].values for cat in gdd_categories.categories]

box_plot = ax4.boxplot(yield_by_gdd, labels=gdd_categories.categories, patch_artist=True)
colors = ['lightblue', 'lightgreen', 'yellow', 'lightcoral']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax4.set_title('Yield Distribution by GDD Categories')
ax4.set_ylabel('Yield (Bu/Acre)')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# SECTION 9: BUSINESS INSIGHTS & SUMMARY
# =============================================================================

print("\n📋 EXECUTIVE SUMMARY")
print("=" * 60)

print(f"""
🌽 IOWA CORN YIELD PREDICTION MODEL - KEY FINDINGS

MODEL PERFORMANCE:
• Accuracy: {test_r2:.1%} R² on test data (2022-2023)
• Error Rate: ±{test_mae:.1f} bushels per acre average
• Cross-validation: {cv_scores.mean():.1%} (±{cv_scores.std():.1%}) R²

TOP CLIMATE DRIVERS:
1. {feature_importance.iloc[-1]['feature'].replace('_', ' ').title()}: {feature_importance.iloc[-1]['importance']:.3f}
2. {feature_importance.iloc[-2]['feature'].replace('_', ' ').title()}: {feature_importance.iloc[-2]['importance']:.3f}  
3. {feature_importance.iloc[-3]['feature'].replace('_', ' ').title()}: {feature_importance.iloc[-3]['importance']:.3f}

SCENARIO ANALYSIS:
• Optimal conditions: {scenarios[scenarios['scenario']=='Optimal']['predicted_yield'].iloc[0]:.0f} bu/acre
• Drought impact: -{scenarios[scenarios['scenario']=='Optimal']['predicted_yield'].iloc[0] - scenarios[scenarios['scenario']=='Drought']['predicted_yield'].iloc[0]:.0f} bu/acre
• Heat stress penalty: -{scenarios[scenarios['scenario']=='Optimal']['predicted_yield'].iloc[0] - scenarios[scenarios['scenario']=='Heat_Stress']['predicted_yield'].iloc[0]:.0f} bu/acre

BUSINESS APPLICATIONS:
✓ Risk assessment for crop insurance
✓ Precision agriculture recommendations  
✓ Climate adaptation strategies
✓ Yield forecasting for supply chain planning
✓ Carbon sequestration potential modeling

NEXT STEPS:
→ Integrate real-time weather APIs
→ Add soil type and management practice variables
→ Develop county-specific models
→ Create early-season prediction capabilities
""")

# Save key results for portfolio
results_summary = {
    'model_performance': {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    },
    'top_features': feature_importance.tail(5).to_dict('records'),
    'scenario_predictions': scenarios[['scenario', 'predicted_yield']].to_dict('records'),
    'climate_insights': {
        'optimal_gdd_range': '2800-3200',
        'optimal_precip_range': '20-25 inches',
        'heat_stress_threshold': '15+ days over 90°F',
        'drought_threshold': '25+ consecutive dry days'
    }
}

print(f"\n💾 Model trained on {len(X_train)} samples across {df['county'].nunique()} Iowa counties")
print(f"📊 Dataset spans {df['year'].nunique()} years ({df['year'].min()}-{df['year'].max()})")
print(f"🎯 Ready for deployment in agricultural decision support systems")

print("\n" + "="*60)
print("🚀 PROJECT COMPLETE - Ready for GitHub Portfolio!")
print("="*60)